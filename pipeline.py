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
from optimization import run_psiblast, sampling, SeqItem, select_flexible_residues
from patchEX_infer import InferenceModel
from utils import map_mutated_residues
from evaluation.structure import sequence_recovery
from evaluation import Evaluator
from typing import Literal
import numpy as np
import argparse
import tempfile
import logging
import random
import torch
import json
import yaml
import os
import shutil
import sys


def resolve_runtime_device(device_preference=None):
    if isinstance(device_preference, torch.device):
        device_preference = str(device_preference)

    if device_preference is None:
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if isinstance(device_preference, str):
        normalized = device_preference.strip().lower()
        if normalized.startswith('cuda'):
            if torch.cuda.is_available():
                return torch.device(device_preference)
            logging.warning(
                "CUDA was requested but is not available. Falling back to CPU."
            )
            return torch.device('cpu')
        if normalized == 'cpu':
            return torch.device('cpu')

    return torch.device(device_preference)


def cleanup_output_artifacts(output_dir, keep_files=None):
    keep_files = set(keep_files or [])

    if not os.path.exists(output_dir):
        return

    for item_name in os.listdir(output_dir):
        if item_name in keep_files:
            continue

        item_path = os.path.join(output_dir, item_name)
        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        except FileNotFoundError:
            pass


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
    save_surrogate_models: bool = False

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
        if 'device' in filtered_dict:
            filtered_dict['device'] = str(resolve_runtime_device(filtered_dict['device']))

        return cls(**filtered_dict)


class IPFPipeline:
    def __init__(self, config_path="MapDiff/conf", config_name="inference", output_dir="PipelineResults", device=None):
        with initialize(version_base=None, config_path=config_path):
            self.cfg = compose(config_name=config_name)

        self.device = resolve_runtime_device(device)

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


        checkpoint = torch.load(self.cfg.test_model.path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.eval()
        enable_dropout(self.model)

        self.collator = CollatorDiff()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir
        self.last_mean_sample_logits = None

    def __call__(self, pdb_file):
        file_name = os.path.basename(pdb_file).split(".")[0]
        save_path = f'{self.output_dir}/{file_name}'
        self.last_mean_sample_logits = None
        try:
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
            self.last_mean_sample_logits = mean_sample_logits
            torch.save(mean_sample_logits, f'{save_path}/ipf_logit.pt')

            true_label = g_batch.x.cpu()
            true_sample_seq = ''.join([amino_acids_type[i] for i in true_label.argmax(dim=1).tolist()])

            ll_fullseq = F.cross_entropy(mean_sample_logits, true_label, reduction='mean').item()
            perplexity = np.exp(ll_fullseq)
            sample_recovery = (mean_sample_logits.argmax(dim=1) == true_label.argmax(dim=1)).sum() / true_label.shape[0]

        except Exception as e:
            raise RuntimeError(f'IPF pipeline failed for {pdb_file}: {e}') from e

        return mean_egnn_feats, true_sample_seq, perplexity, float(sample_recovery)


class Pipeline:
    def __init__(self, config):
        self.config = config
        self.task = config['task']
        self.device = resolve_runtime_device(config.get('device', config.get('de_config', {}).get('device')))
        self.output_dir = config['IPF_config']['output_dir']
        ipf_config = dict(config['IPF_config'])
        ipf_config.setdefault('device', str(self.device))
        self.ipf_pipeline = IPFPipeline(**ipf_config)
        self.config.setdefault('de_config', {})['device'] = str(self.device)
        oracle_model_config = dict(config['oracle_model'])
        oracle_model_config.setdefault('device', str(self.device))
        self.oracle_model = InferenceModel(**oracle_model_config)
        self.evaluator = Evaluator(task=self.task)

    def __call__(self, pdb_file, ec_pool, target_value):
        pdb_accession = os.path.basename(pdb_file).split(".")[0]
        accession = pdb_accession
        ipf_res = os.path.join(self.output_dir, accession)
        os.makedirs(ipf_res, exist_ok=True)
        cleanup_output_artifacts(ipf_res)
        final_results = None
        print(f"\n[INFO] Processing: accession={accession}, ec_pool={ec_pool}, target={target_value}")

        mean_egnn_feats, true_seq, perplexity, sample_recovery = self.ipf_pipeline(pdb_file)
        ipf_logits = self.ipf_pipeline.last_mean_sample_logits
        if ipf_logits is None:
            raise RuntimeError(f"IPF logits were not generated for {accession}")

        with tempfile.TemporaryDirectory(prefix=f'{accession}_', dir=self.output_dir) as temp_dir:
            ipf_fasta = os.path.join(temp_dir, f'{accession}.fasta')
            with open(ipf_fasta, 'w') as fasta_file:
                fasta_file.write(f'>{accession}\n{true_seq}\n')

            pssm = run_psiblast(ipf_fasta, ec_pool, threads=8, show_realtime=True, output_dir=temp_dir)
            print(pssm)
            if not pssm or not os.path.exists(pssm) or os.path.getsize(pssm) == 0:
                raise RuntimeError(f"PSI-BLAST failed for {accession} in EC {ec_pool}")

            pssm_report = os.path.join(os.path.dirname(pssm), 'psiblast_report.txt')
            shutil.copy2(pssm, os.path.join(ipf_res, 'seed.pssm'))
            if os.path.exists(pssm_report):
                shutil.copy2(pssm_report, os.path.join(ipf_res, 'psiblast_report.txt'))

            evolutionary_backbone_logits, accessions, evolutionary_backbone_seqs = sampling(ipf_logits, pssm, accession, k=self.config['sampling']['n'],)

        np.save(os.path.join(ipf_res, 'evolutionary_backbone_logits.npy'), evolutionary_backbone_logits)

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
            # Reset all loggers cleanly
            for logger_name in ['main', 'directed_evolution']:
                logger_temp = logging.getLogger(logger_name)
                for handler in logger_temp.handlers[:]:
                    handler.close()
                    logger_temp.removeHandler(handler)

            root_logger = logging.getLogger()
            root_logger.handlers.clear()
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s'
            ))
            root_logger.addHandler(stream_handler)
            root_logger.setLevel(logging.WARNING)

            logger = logging.getLogger('main')
            logger.setLevel(logging.WARNING)
            logger.info("START")

            try:
                for i in range(1, config_de.run_times + 1):
                    results = de.run(config_de)
                    best_idx = int(np.argmax(results['pas']))
                    mutated_residues = results['sequence'][best_idx]
                    best_pas = results['pas'][best_idx]
                    mapped_seq = map_mutated_residues(selected_residue_idx, [mutated_residues], wt_seq)[0]
                    seq_rec = sequence_recovery(true_seq, mapped_seq)

                    final_results = {
                        'accession': accession,
                        'ec': os.path.basename(ec_pool).strip('.fasta'),
                        'target_value': target_value,
                        'sequence': mapped_seq,
                        'perplexity': perplexity,
                        'mutated_residues': mutated_residues,
                        'selected_residue_idx': selected_residue_idx,
                        'sequence_recovery': seq_rec,
                        'PAS': best_pas,
                        'output_dir': ipf_res
                    }
                    try:
                        metric = self.evaluator(final_results, reference_pdb=pdb_file)
                        final_results.update(metric)
                    except Exception as eval_ex:
                        logger.error(f"Evaluation failed: {eval_ex}")

                    with open(f"{ipf_res}/report.json", 'w') as f:
                        json.dump(final_results, f, indent=4)

                    with open(f"{ipf_res}/{accession}.fasta", 'w') as f:
                        f.write(f'>{accession}\n{mapped_seq}\n')

            except Exception as run_ex:
                print(run_ex)

            final_output_files = {'report.json', 'ipf_logit.pt', 'seed.pssm', 'psiblast_report.txt', 'evolutionary_backbone_logits.npy'}
            final_output_files.update(
                file_name for file_name in os.listdir(ipf_res) if file_name.endswith('.fasta')
            )
            final_output_files.update(
                file_name for file_name in os.listdir(ipf_res) if file_name.endswith('_pred.pdb')
            )
            cleanup_output_artifacts(ipf_res, keep_files=final_output_files)

        if final_results is None:
            raise RuntimeError(f'No final results were produced for {accession}')

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
