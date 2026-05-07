from hydra import compose, initialize_config_dir
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
from typing import Literal
import numpy as np
import argparse
import copy
import importlib.util
import tempfile
import logging
import random
import torch
import json
import yaml
import os
import shutil
import sys
from setup_assets import ensure_assets, AssetSetupError, build_asset_targets
from pathlib import Path


def load_ped_eval_single_sequence_api():
    try:
        from ped_eval import evaluate_single_sequence as ped_eval_single_sequence
        return ped_eval_single_sequence
    except ImportError:
        ped_eval_init = os.path.join(
            os.path.dirname(__file__),
            "PED-Eval",
            "ped_eval",
            "__init__.py",
        )
        spec = importlib.util.spec_from_file_location(
            "ped_eval",
            ped_eval_init,
            submodule_search_locations=[os.path.dirname(ped_eval_init)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load ped_eval from {ped_eval_init}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["ped_eval"] = module
        spec.loader.exec_module(module)
        return module.evaluate_single_sequence


evaluate_single_sequence = load_ped_eval_single_sequence_api()
# ESMFold metrics are optional; sequence_recovery is always expected.
PIPELINE_REPORT_METRICS = ("sequence_recovery", "mean_plddt", "tm_score", "rmsd")
REPO_ROOT = Path(__file__).resolve().parent


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


def resolve_repo_relative_path(path_value, *, repo_root: Path = REPO_ROOT, field_name: str = "path") -> str:
    if path_value is None or str(path_value).strip() == "":
        raise ValueError(f"Expected a valid non-empty path for {field_name}.")
    resolved_path = Path(path_value)
    if resolved_path.is_absolute():
        return str(resolved_path)
    return str((repo_root / resolved_path).resolve())


def resolve_runtime_path(path_value, *, repo_root: Path = REPO_ROOT, field_name: str = "path") -> str:
    if path_value is None or str(path_value).strip() == "":
        raise ValueError(f"Expected a valid non-empty path for {field_name}.")

    raw_path = Path(path_value).expanduser()
    if raw_path.is_absolute():
        return str(raw_path.resolve())

    cwd_candidate = (Path.cwd() / raw_path).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    return str((repo_root / raw_path).resolve())


def is_repo_data_design_path(path_value, *, repo_root: Path = REPO_ROOT) -> bool:
    resolved_path = Path(resolve_runtime_path(path_value, repo_root=repo_root))
    try:
        relative_path = resolved_path.relative_to(repo_root)
    except ValueError:
        return False
    return bool(relative_path.parts) and relative_path.parts[0] == 'data_design'


def normalize_pipeline_config_paths(config: dict, *, repo_root: Path = REPO_ROOT) -> dict:
    normalized_config = copy.deepcopy(config)

    ipf_config = normalized_config.setdefault('IPF_config', {})
    if 'config_path' in ipf_config:
        ipf_config['config_path'] = resolve_repo_relative_path(
            ipf_config['config_path'],
            repo_root=repo_root,
            field_name='IPF_config.config_path',
        )
    if 'output_dir' in ipf_config:
        ipf_config['output_dir'] = resolve_repo_relative_path(
            ipf_config['output_dir'],
            repo_root=repo_root,
            field_name='IPF_config.output_dir',
        )

    oracle_model_config = normalized_config.setdefault('oracle_model', {})
    if 'checkpoint_dir' in oracle_model_config:
        oracle_model_config['checkpoint_dir'] = resolve_repo_relative_path(
            oracle_model_config['checkpoint_dir'],
            repo_root=repo_root,
            field_name='oracle_model.checkpoint_dir',
        )

    return normalized_config


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


def ensure_runtime_assets(*, pdb_path, ec_pool_path, repo_root: Path = REPO_ROOT) -> None:
    asset_targets = build_asset_targets(repo_root)
    requested_bundles: list[str] = []

    pipeline_target_names = ('mapdiff_weight', 'patchex_weight', 'esm150')
    if any(not asset_targets[target_name].is_installed() for target_name in pipeline_target_names):
        requested_bundles.append('pipeline')

    needs_repo_data = (
        is_repo_data_design_path(pdb_path, repo_root=repo_root)
        or is_repo_data_design_path(ec_pool_path, repo_root=repo_root)
    )
    if needs_repo_data and not asset_targets['data_design'].is_installed():
        requested_bundles.insert(0, 'ped-eval')

    if requested_bundles:
        ensure_assets(bundles=tuple(requested_bundles), repo_root=repo_root)


def require_existing_file(path_value, *, description: str) -> str:
    resolved_path = Path(path_value)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved_path}")
    return str(resolved_path)


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
    def __init__(
        self,
        config_path="MapDiff/conf",
        config_name="inference",
        output_dir="PipelineResults",
        device=None,
        repo_root=None,
    ):
        self.repo_root = Path(repo_root or REPO_ROOT).resolve()
        resolved_config_path = resolve_repo_relative_path(
            config_path,
            repo_root=self.repo_root,
            field_name='IPF_config.config_path',
        )

        with initialize_config_dir(version_base=None, config_dir=resolved_config_path):
            self.cfg = compose(config_name=config_name)

        self.device = resolve_runtime_device(device)
        marginal_dist_path = resolve_repo_relative_path(
            self.cfg.dataset.marginal_train_dir,
            repo_root=self.repo_root,
            field_name='dataset.marginal_train_dir',
        )
        test_model_path = resolve_repo_relative_path(
            self.cfg.test_model.path,
            repo_root=self.repo_root,
            field_name='test_model.path',
        )

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
                                marginal_dist_path=marginal_dist_path).to(self.device)


        checkpoint = torch.load(test_model_path, map_location='cpu')
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
        self.config = normalize_pipeline_config_paths(config)
        self.task = self.config['task']
        self.device = resolve_runtime_device(
            self.config.get('device', self.config.get('de_config', {}).get('device'))
        )
        self.output_dir = self.config['IPF_config']['output_dir']
        ipf_config = copy.deepcopy(self.config['IPF_config'])
        ipf_config.setdefault('device', str(self.device))
        ipf_config.setdefault('repo_root', str(REPO_ROOT))
        self.ipf_pipeline = IPFPipeline(**ipf_config)
        self.config.setdefault('de_config', {})['device'] = str(self.device)
        oracle_model_config = copy.deepcopy(self.config['oracle_model'])
        oracle_model_config.setdefault('device', str(self.device))
        self.oracle_model = InferenceModel(**oracle_model_config)

    def __call__(self, pdb_file, ec_pool, target_value):
        pdb_file = require_existing_file(
            resolve_runtime_path(pdb_file, repo_root=REPO_ROOT, field_name='pdb_file'),
            description='PDB file',
        )
        ec_pool = require_existing_file(
            resolve_runtime_path(ec_pool, repo_root=REPO_ROOT, field_name='ec_pool'),
            description='EC pool file',
        )
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
                    evaluation_metrics = {}

                    final_results = {
                        'accession': accession,
                        'ec': os.path.basename(ec_pool).strip('.fasta'),
                        'target_value': target_value,
                        'sequence': mapped_seq,
                        'perplexity': perplexity,
                        'mutated_residues': mutated_residues,
                        'selected_residue_idx': selected_residue_idx,
                        'PAS': best_pas,
                        'output_dir': ipf_res
                    }
                    try:
                        evaluation_metrics = evaluate_single_sequence(
                            accession=accession,
                            predicted_sequence=mapped_seq,
                            native_sequence=true_seq,
                            reference_pdb=pdb_file,
                            use_esmfold=True,
                            esmfold_output_dir=ipf_res,
                        )
                    except Exception as eval_ex:
                        logger.error(f"Evaluation failed: {eval_ex}")
                    for metric_name in PIPELINE_REPORT_METRICS:
                        if metric_name in evaluation_metrics:
                            final_results[metric_name] = evaluation_metrics[metric_name]

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
    parser.add_argument(
        '--skip-asset-setup',
        action='store_true',
        help='skip automatic download/install of missing pipeline weights and repo data assets before execution',
    )
    args = parser.parse_args()

    args.config = resolve_runtime_path(args.config, repo_root=REPO_ROOT, field_name='config')
    args.pdb = resolve_runtime_path(args.pdb, repo_root=REPO_ROOT, field_name='pdb')
    args.ec_pool = resolve_runtime_path(args.ec_pool, repo_root=REPO_ROOT, field_name='ec_pool')

    if not args.skip_asset_setup:
        try:
            ensure_runtime_assets(
                pdb_path=args.pdb,
                ec_pool_path=args.ec_pool,
                repo_root=REPO_ROOT,
            )
        except AssetSetupError as exc:
            print(f'Automatic asset setup failed: {exc}', file=sys.stderr)
            sys.exit(1)

    try:
        args.config = require_existing_file(args.config, description='Config file')
        args.pdb = require_existing_file(args.pdb, description='PDB file')
        args.ec_pool = require_existing_file(args.ec_pool, description='EC pool file')
    except FileNotFoundError as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        sys.exit(1)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    pipeline = Pipeline(config)
    res = pipeline(args.pdb, args.ec_pool, args.target_value)
    for k, v in res.items():
        print(f"{k}: {v}")