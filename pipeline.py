from hydra import initialize, compose
import direct_evolution.directed_evolution as de
from MapDiff.model.egnn_pytorch.egnn_net import EGNN_NET
from MapDiff.model.ipa.ipa_net import IPANetPredictor
from MapDiff.model.prior_diff import Prior_Diff
from MapDiff.utils import enable_dropout
from MapDiff.dataloader.collator import CollatorDiff
from MapDiff.data.generate_graph_cath import pdb2graph, get_processed_graph, amino_acids_type
import torch.nn.functional as F
from dataclasses import dataclass
from optimization import extract_pred_to_fasta, run_psiblast, sampling, SeqItem, select_flexible_residues
from patchEX_infer import InferenceModel
from utils import map_mutated_residues
from evaluation.structure import sequence_recovery
from evaluation import Evaluator
from typing import Literal
import pandas as pd
import numpy as np
import argparse
import logging
import random
import torch
import json
import yaml
import os


@dataclass
class ConfigDe:
    wt_accession: str
    selected_residue_idx: list
    wt_seq: str
    oracle_model: InferenceModel
    target_value: float
    de_out: str
    seed: int = 42
    esm: Literal["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"] = "esm2_t33_650M_UR50D"
    task: Literal["GB1", "PhoQ"] = "opt"
    evo_steps: int = 10
    max_oracle_call_per_step: int = 100
    max_mutation: int = 1
    max_sample_steps: int = 100
    internal_steps: int = 64
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    oracle_batch_size: int = 16
    parallel_samples: int = 128
    lr: float = 0.001
    oracle: Literal["wetlab"] = "wetlab"
    dropout: float = 0.1
    embed_dim: int = 256
    hidden_dim: int = 256
    top_k: int = 100
    patience: int = 3
    eta: float = 0.1
    ensemble: int = 3
    run_times: int = 1
    sampler: Literal["lmc", "hmc", "random"] = "hmc"
    exp_name: str = "default"
    onehot_constraint: int = 1

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ConfigDe':
        """
        Load configuration from a dictionary into ConfigDe class.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            ConfigDe instance with loaded configuration
        """
        # Filter only the fields that exist in the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}

        return cls(**filtered_dict)


class IPFPipeline:
    def __init__(self, config_path="MapDiff/conf", config_name="inference", output_dir="PipelineResults"):
        with initialize(version_base=None, config_path=config_path):
            self.cfg = compose(config_name=config_name)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # load trained model
        egnn = EGNN_NET(input_feat_dim=self.cfg.model.input_feat_dim, hidden_channels=self.cfg.model.hidden_dim,
                        edge_attr_dim=self.cfg.model.edge_attr_dim,
                        dropout=self.cfg.model.drop_out, n_layers=self.cfg.model.depth, update_edge=self.cfg.model.update_edge,
                        norm_coors=self.cfg.model.norm_coors, update_coors=self.cfg.model.update_coors,
                        update_global=self.cfg.model.update_global, embedding=self.cfg.model.embedding,
                        embedding_dim=self.cfg.model.embedding_dim, norm_feat=self.cfg.model.norm_feat, embed_ss=self.cfg.model.embed_ss)

        ipa = IPANetPredictor(dropout=self.cfg.model.ipa_drop_out)

        self.model = Prior_Diff(egnn, ipa, timesteps=self.cfg.diffusion.timesteps,
                                objective=self.cfg.diffusion.objective,
                                noise_type=self.cfg.diffusion.noise_type, sample_method=self.cfg.diffusion.sample_method,
                                min_mask_ratio=self.cfg.mask_prior.min_mask_ratio,
                                dev_mask_ratio=self.cfg.mask_prior.dev_mask_ratio,
                                marginal_dist_path=self.cfg.dataset.marginal_train_dir).to(self.device)


        checkpoint = torch.load(self.cfg.test_model.path)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.eval()
        enable_dropout(self.model)

        self.collator = CollatorDiff()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def __call__(self, pdb_file):
        try:
            file_name = os.path.basename(pdb_file).split(".")[0]
            save_path = f'{self.output_dir}/{file_name}'
            os.makedirs(save_path, exist_ok=True)
            graph = get_processed_graph(pdb2graph(pdb_file))
            g_batch, ipa_batch = self.collator([graph])
            g_batch = g_batch.to(self.device)
            ipa_batch = ipa_batch.to(self.device)

            ens_logits = []
            egnn_feats_list = []
            with torch.no_grad():
                for _ in range(self.cfg.diffusion.ensemble_num):
                    logits, sample_graph = self.model.mc_ddim_sample(g_batch, ipa_batch, diverse=True, step=self.cfg.diffusion.ddim_steps)
                    egnn_feats = sample_graph[0]
                    egnn_feats_list.append(egnn_feats)
                    ens_logits.append(logits)

            ens_logits_tensor = torch.stack(ens_logits)
            mean_sample_logits = ens_logits_tensor.mean(dim=0).cpu()
            mean_egnn_feats = torch.stack(egnn_feats_list).mean(dim=0).cpu()
            torch.save(mean_sample_logits, f'{save_path}/logits.pt')
            torch.save(mean_egnn_feats, f'{save_path}/egnn_feats.pt')

            true_label = g_batch.x.cpu()
            true_sample_seq = ''.join([amino_acids_type[i] for i in true_label.argmax(dim=1).tolist()])
            pred_sample_seq = ''.join([amino_acids_type[i] for i in mean_sample_logits.argmax(dim=1).tolist()])

            ll_fullseq = F.cross_entropy(mean_sample_logits, true_label, reduction='mean').item()
            perplexity = np.exp(ll_fullseq)
            sample_recovery = (mean_sample_logits.argmax(dim=1) == true_label.argmax(dim=1)).sum() / true_label.shape[0]

            with open(f'{save_path}/pred.txt', 'w') as f:
                f.write(f'>pred\n{pred_sample_seq}\n')
                f.write(f'>true\n{true_sample_seq}\n')
                f.write(f'Sequence length: {len(pred_sample_seq)}\n')
                f.write(f'Sample perplexity: {perplexity}\n')
                f.write(f'Sample recovery rate {sample_recovery:.4f}\n')

        except Exception as e:
            print(repr(e))

        return save_path, true_sample_seq


class Pipeline:
    def __init__(self, config):
        self.config = config
        self.task = config['task']
        self.output_dir = config['IPF_config']['output_dir']
        self.ipf_pipeline = IPFPipeline(**config['IPF_config'])
        self.oracle_model = InferenceModel(**config['oracle_model'])
        self.evaluator = Evaluator(task=self.task)

    def __call__(self, pdb_file, ec_pool, target_value):
        accession = os.path.basename(pdb_file).split(".")[0]
        print(f"\n[INFO] Processing: accession={accession}, ec_pool={ec_pool}, target={target_value}")

        ipf_res, true_seq = self.ipf_pipeline(pdb_file)

        ipf_fasta = extract_pred_to_fasta(f'{ipf_res}/pred.txt')
        pssm = run_psiblast(ipf_fasta, ec_pool, threads=8, show_realtime=True, output_dir=self.output_dir)
        print(pssm)
        if not pssm or not os.path.exists(pssm) or os.path.getsize(pssm) == 0:
            raise RuntimeError(f"PSI-BLAST failed for {accession} in EC {ec_pool}")
        ipf_logits = f'{ipf_res}/logits.pt'
        evolutionary_backbone_logits, accessions, evolutionary_backbone_seqs = sampling(ipf_logits, pssm, accession, k=self.config['sampling']['n'],)
        np.save(f'{ipf_res}/evolutionary_backbone_logits.npy', evolutionary_backbone_logits)
        seq_data = [SeqItem(accession=acc, sequence=seq) for acc, seq in zip(accessions, evolutionary_backbone_seqs)]
        seq_data = self.oracle_model.inference(seq_data)
        for item in seq_data:
            item.map_weights()

        for item in seq_data:
            item.idx, item.score = select_flexible_residues(evolutionary_backbone_logits, item.weights, k=self.config['sampling']['k'], beta=0.5, exclude_idx=None)
            print(f'Selected positions for {item.accession}: {item.idx}')

        for item in seq_data:
            accession = item.accession
            selected_residue_idx = item.idx
            wt_seq = item.sequence

            self.config['de_config']['wt_accession'] = accession
            self.config['de_config']['selected_residue_idx'] = selected_residue_idx
            self.config['de_config']['wt_seq'] = wt_seq
            self.config['de_config']['oracle_model'] = self.oracle_model
            self.config['de_config']['target_value'] = target_value
            self.config['de_config']['task'] = self.task
            self.config['de_config']['de_out'] = ipf_res
            config_de = ConfigDe.from_dict(self.config['de_config'])
            seq_info = {'accession': accession, 'seq': wt_seq, 'selected_idx': selected_residue_idx}
            with open(f"{ipf_res}/seq_info.json", 'w') as f:
                json.dump(seq_info, f, indent=4)

            log_filename = f"{ipf_res}/log_{config_de.exp_name}.txt"
            # Reset all loggers cleanly
            for logger_name in ['main', 'directed_evolution']:
                logger_temp = logging.getLogger(logger_name)
                for handler in logger_temp.handlers[:]:
                    handler.close()
                    logger_temp.removeHandler(handler)

            root_logger = logging.getLogger()
            root_logger.handlers.clear()
            file_handler = logging.FileHandler(log_filename, mode='w')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s'
            ))
            root_logger.addHandler(file_handler)
            root_logger.setLevel(logging.INFO)

            logger = logging.getLogger('main')
            logger.setLevel(logging.INFO)
            logger.info("START")

            try:
                data_filename = f'{ipf_res}/results.xlsx'
                with pd.ExcelWriter(data_filename, engine='xlsxwriter') as writer:
                    for i in range(1, config_de.run_times + 1):
                        results = de.run(config_de)
                        df = pd.DataFrame(results)
                        df.to_excel(writer, sheet_name=f'{i}', index=False)

                        best_row = df.loc[df['fitness'].idxmax()]
                        mutated_residues = best_row['sequence']
                        mapped_seq = map_mutated_residues(selected_residue_idx, [mutated_residues], wt_seq)[0]
                        seq_rec = sequence_recovery(true_seq, mapped_seq)

                        final_results = {
                            'accession': accession,
                            'ec': os.path.basename(ec_pool).strip('.fasta'),
                            'target_value': target_value,
                            'sequence': mapped_seq,
                            'mutated_residues': mutated_residues,
                            'selected_residue_idx': selected_residue_idx,
                            'sequence_recovery': seq_rec,
                            'fitness': best_row['fitness'],
                            'output_dir': ipf_res
                        }
                        try:
                            metric = self.evaluator(final_results, reference_pdb=pdb_file)
                            final_results.update(metric)
                        except Exception as eval_ex:
                            logger.error(f"Evaluation failed: {eval_ex}")

                        with open(f"{ipf_res}/result.json", 'w') as f:
                            json.dump(final_results, f, indent=4)

            except Exception as run_ex:
                print(run_ex)

        print(f'[INFO] Finished processing: accession={accession}, ec_pool={ec_pool}, target={target_value}')
        return final_results


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    setup_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='pipeline config to use')
    parser.add_argument('--pdb', type=str, required=True, help='input pdb file')
    parser.add_argument('--ec_pool', type=str, required=True, help='ec fasta file for psiblast')
    parser.add_argument('--target_value', type=float, required=True, help='target value for optimization')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    pipeline = Pipeline(config)
    res = pipeline(args.pdb, args.ec_pool, args.target_value)
    for k, v in res.items():
        print(f"{k}: {v}")
