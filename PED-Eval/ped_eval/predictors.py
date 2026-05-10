from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union


_ALLOWED_BY_TASK = {
    "opt": {"Seq2Topt", "PatchEX", "PatchET"},
    "ph": {"Seq2pHopt", "PatchEX", "EpHod"},
}


@dataclass
class _SeqItem:
    accession: str
    sequence: str
    label: float = 0.0


def allowed_predictors(task: str) -> List[str]:
    t = str(task).lower()
    if t not in _ALLOWED_BY_TASK:
        raise ValueError("task must be 'opt' or 'ph'")
    return sorted(_ALLOWED_BY_TASK[t])


def validate_predictor_for_task(task: str, predictor: str) -> None:
    allowed = set(allowed_predictors(task))
    if predictor not in allowed:
        raise ValueError(
            f"predictor '{predictor}' is not valid for task '{task}'. "
            f"Allowed: {sorted(allowed)}"
        )


class BasePredictor:
    def predict_many(self, sequences: Mapping[str, str]) -> Dict[str, Optional[float]]:
        raise NotImplementedError


def _first_existing_path(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _load_seq2_model_class():
    """Return Seq2Topt MultiAttModel with checkpoint-compatible parameter names."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class RDBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dense = nn.Linear(dim, dim)

        def forward(self, x):
            x0 = x
            x = F.leaky_relu(self.dense(x))
            return x0 + x

    class MultiAttModel(nn.Module):
        def __init__(self, dim, window, n_head, n_rd):
            super().__init__()
            self.n_RD = n_rd
            self.n_head = n_head
            self.cnn_v = nn.Conv1d(dim, dim, kernel_size=2 * window + 1, padding=window)
            # IMPORTANT: keep original attribute names for checkpoint compatibility.
            self.W_cnns = nn.ModuleList(
                [nn.Conv1d(dim, dim, kernel_size=2 * window + 1, padding=window) for _ in range(n_head)]
            )
            self.RDs = nn.ModuleList([RDBlock(2 * n_head * dim) for _ in range(n_rd)])
            self.output = nn.Linear(2 * n_head * dim, 1)

        def forward(self, emb):
            values = self.cnn_v(emb)
            for i in range(self.n_head):
                weights = F.softmax(self.W_cnns[i](emb), dim=-1)
                x_sum = torch.sum(values * weights, dim=-1)
                x_max, _ = torch.max(values * weights, dim=-1)
                if i == 0:
                    cat_xsum = x_sum
                    cat_xmax = x_max
                else:
                    cat_xsum = torch.cat([cat_xsum, x_sum], dim=1)
                    cat_xmax = torch.cat([cat_xmax, x_max], dim=1)
            cat_f = torch.cat([cat_xsum, cat_xmax], dim=1)
            for i in range(self.n_RD):
                cat_f = self.RDs[i](cat_f)
            return self.output(cat_f)

    return MultiAttModel, torch


class Seq2RegressorPredictor(BasePredictor):
    def __init__(self, model_path: Path, value_scale: float, batch_size: int = 4) -> None:
        try:
            import esm
        except Exception as exc:
            raise ImportError("Seq2Topt/Seq2pHopt predictor requires the 'esm' package.") from exc

        MultiAttModel, torch = _load_seq2_model_class()
        self._torch = torch
        self._esm = esm
        self.batch_size = int(batch_size)
        self.value_scale = float(value_scale)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not model_path.exists():
            raise FileNotFoundError(f"Seq2 checkpoint not found: {model_path}")

        model = MultiAttModel(320, 3, 4, 4).to(self.device)
        state = torch.load(str(model_path), map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        self.model = model

        esm2_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.esm2_model = esm2_model.to(self.device).eval()
        self.batch_converter = alphabet.get_batch_converter()

    def predict_many(self, sequences: Mapping[str, str]) -> Dict[str, Optional[float]]:
        ids = list(sequences.keys())
        seqs = [str(sequences[k]).strip().upper() for k in ids]

        valid_idx = [i for i, s in enumerate(seqs) if len(s) > 0]
        out: Dict[str, Optional[float]] = {k: None for k in ids}
        if not valid_idx:
            return out

        valid_pairs = [(ids[i], seqs[i]) for i in valid_idx]
        for start in range(0, len(valid_pairs), self.batch_size):
            chunk = valid_pairs[start : start + self.batch_size]
            labels, strings, batch_tokens = self.batch_converter(chunk)
            _ = labels, strings
            batch_tokens = batch_tokens.to(device=self.device, non_blocking=True)

            with self._torch.no_grad():
                emb = self.esm2_model(batch_tokens, repr_layers=[6], return_contacts=False)
            emb = emb["representations"][6].transpose(1, 2).to(self.device)

            with self._torch.no_grad():
                preds = self.model(emb).cpu().detach().numpy().reshape(-1)

            for (acc, _seq), pred in zip(chunk, preds):
                out[acc] = float(pred * self.value_scale)

        return out


class PatchInferencePredictor(BasePredictor):
    def __init__(
        self,
        model_config_path: Path,
        weight_path: Path,
        project_root: Path,
        batch_size: int = 4,  # FIX: accept batch_size
    ) -> None:
        try:
            import torch
            import yaml
        except Exception as exc:
            raise ImportError("PatchEX/PatchET predictor requires torch and pyyaml.") from exc

        repo_root = project_root.resolve()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        try:
            from models import load_model
            from utils import load_weight
        except Exception as exc:
            raise ImportError("Failed to import root 'models' or 'utils' modules for Patch predictors.") from exc

        if not model_config_path.exists():
            raise FileNotFoundError(f"PatchEX config not found: {model_config_path}")
        if not weight_path.exists():
            raise FileNotFoundError(f"PatchEX weight not found: {weight_path}")

        config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
        config = self._normalize_config_paths(config, repo_root)

        Model, Collator = load_model(config["name"])
        self.collator = Collator(config["pretrain_model"])
        try:
            self.model = Model(config)
        except NotADirectoryError as exc:
            # Defensive fallback for configs where pretrained_patchet points to a YAML file
            # but downstream code accidentally treats it as a directory.
            msg = str(exc)
            pretrained_patchet = str(config.get("pretrained_patchet", "")).strip()
            if pretrained_patchet and "model_config.yaml" in msg:
                normalized = pretrained_patchet.strip().strip('"').strip("'").rstrip("/\\")
                normalized = normalized.replace("\\", "/")
                p = Path(normalized)
                if not p.is_absolute():
                    p = (repo_root / p).resolve()
                if p.suffix.lower() == ".yml":
                    p = p.with_suffix(".yaml")
                config["pretrained_patchet"] = p.as_posix()
                self.model = Model(config)
            else:
                raise
        load_weight(self.model, str(weight_path))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if hasattr(self.model, "inference"):
            self.model.inference = True
        self.model.eval()
        self._torch = torch
        self.batch_size = int(batch_size)  # FIX: store batch_size

    @staticmethod
    def _normalize_config_paths(config: dict, project_root: Path) -> dict:
        cfg = dict(config)

        pretrain_model = cfg.get("pretrain_model")
        if isinstance(pretrain_model, str) and pretrain_model:
            p = Path(pretrain_model)
            if not p.is_absolute() and (project_root / p).exists():
                cfg["pretrain_model"] = str((project_root / p).resolve())
            elif pretrain_model == "esm150":
                # Fallback to HF model id when local esm150 directory is not available.
                cfg["pretrain_model"] = "facebook/esm2_t30_150M_UR50D"

        pretrained_patchet = cfg.get("pretrained_patchet")
        if isinstance(pretrained_patchet, str) and pretrained_patchet:
            cleaned = pretrained_patchet.strip().strip('"').strip("'")
            cleaned = cleaned.rstrip("/\\").replace("\\", "/")
            p = Path(cleaned)
            if not p.is_absolute():
                p = (project_root / p).resolve()
            # Normalize *.yml to *.yaml so downstream endswith("yaml") checks are stable.
            if p.suffix.lower() == ".yml":
                p = p.with_suffix(".yaml")
            # Keep a stable normalized string so Model(config) can detect *.yaml reliably.
            cfg["pretrained_patchet"] = p.as_posix().strip()

        return cfg

    def predict_many(self, sequences: Mapping[str, str]) -> Dict[str, Optional[float]]:
        items = [_SeqItem(accession=k, sequence=str(v).strip().upper()) for k, v in sequences.items()]
        out: Dict[str, Optional[float]] = {it.accession: None for it in items}
        valid_items = [it for it in items if len(it.sequence) > 0]
        if not valid_items:
            return out

        # FIX: process in chunks instead of one giant batch
        for start in range(0, len(valid_items), self.batch_size):
            chunk = valid_items[start : start + self.batch_size]

            batch = self.collator(chunk)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with self._torch.no_grad():
                outputs = self.model(**batch)

            for i, item in enumerate(chunk):
                pred = outputs.pred[i]
                out[item.accession] = float(pred.item() if hasattr(pred, "item") else pred)

        return out


class EpHodPredictor(BasePredictor):
    def __init__(self, project_root: Path) -> None:
        try:
            import esm
            import joblib
            import numpy as np
            import torch
            import torch.nn as nn
        except Exception as exc:
            raise ImportError("EpHod predictor requires torch, esm, joblib, and numpy.") from exc

        class _ResidualDense(nn.Module):
            def __init__(self, dim=2560, dropout=0.0):
                super().__init__()
                self.dense = nn.Linear(dim, dim)
                self.batchnorm = nn.BatchNorm1d(dim)
                self.activation = nn.ELU()
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                x0 = x
                x = self.dense(x)
                x = self.batchnorm(x)
                x = self.activation(x)
                x = self.dropout(x)
                return x0 + x

        class _LightAttention(nn.Module):
            def __init__(self, dim=1280, kernel_size=7):
                super().__init__()
                samepad = kernel_size // 2
                self.values_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=samepad)
                self.weights_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=samepad)
                self.softmax = nn.Softmax(dim=-1)

            def forward(self, x, mask=None):
                if mask is None:
                    mask = torch.ones(x.shape[0], x.shape[2], dtype=torch.int32, device=x.device)
                values = self.values_conv(x).masked_fill(mask[:, None, :] == 0, -1e6)
                weights = self.weights_conv(x).masked_fill(mask[:, None, :] == 0, -1e6)
                weights = self.softmax(weights)
                x_sum = torch.sum(values * weights, dim=-1)
                x_max, _ = torch.max(values, dim=-1)
                return torch.cat([x_sum, x_max], dim=1), weights

        class _ResidualLightAttention(nn.Module):
            def __init__(self, dim=1280, kernel_size=7, dropout=0.0, res_blocks=4):
                super().__init__()
                self.light_attention = _LightAttention(dim, kernel_size)
                self.batchnorm = nn.BatchNorm1d(2 * dim)
                self.dropout = nn.Dropout(dropout)
                self.residual_dense = nn.ModuleList([_ResidualDense(2 * dim, dropout) for _ in range(res_blocks)])
                self.output = nn.Linear(2 * dim, 1)

            def forward(self, x, mask=None):
                x, weights = self.light_attention(x, mask)
                x = self.batchnorm(x)
                x = self.dropout(x)
                for layer in self.residual_dense:
                    x = layer(x)
                y = self.output(x).flatten()
                return [y, x, weights]

        class _EpHodModel:
            def __init__(self, weights_root: Path):
                import random

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)

                self.esm1v_model, esm1v_alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
                self.esm1v_batch_converter = esm1v_alphabet.get_batch_converter()
                self.esm1v_model = self.esm1v_model.to(self.device)

                self.rlat_model = _ResidualLightAttention(dim=1280, kernel_size=7, dropout=0.0, res_blocks=4).to(self.device)
                local_rlat = _first_existing_path(
                    (weights_root / "Ephod" / "ESM1v-RLATtr.pt").resolve(),
                    (weights_root / "ephod" / "ESM1v-RLATtr.pt").resolve(),
                )
                if local_rlat.exists():
                    model_dict = torch.load(str(local_rlat), map_location=self.device)
                else:
                    url = "https://zenodo.org/records/14252615/files/ESM1v-RLATtr.pt?download=1"
                    model_dict = torch.hub.load_state_dict_from_url(url, progress=False, map_location=self.device)
                model_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in model_dict.items()}
                self.rlat_model.load_state_dict(model_dict)

                svr_candidates = [
                    (weights_root / "ephod" / "ESM1v-SVR.pkl").resolve(),
                    (weights_root / "Ephod" / "ephod" / "data" / "ESM1v-SVR.pkl").resolve(),
                    (weights_root / "ephod" / "data" / "ESM1v-SVR.pkl").resolve(),
                ]
                svr_path = next((p for p in svr_candidates if p.exists()), None)
                if svr_path is None:
                    raise FileNotFoundError(
                        "Missing EpHod SVR file. Expected one of: " + ", ".join(p.as_posix() for p in svr_candidates)
                    )
                self.svr_model, self.svr_stats = joblib.load(str(svr_path))

                self.esm1v_model.eval()
                self.rlat_model.eval()

            @staticmethod
            def _replace_noncanonical(seq: str, replace_char: str = "X") -> str:
                for char in ["B", "J", "O", "U", "Z"]:
                    seq = seq.replace(char, replace_char)
                return seq

            def predict(self, accs, seqs):
                seqs = [self._replace_noncanonical(seq, "X") for seq in seqs]
                data = [(accs[i], seqs[i]) for i in range(len(accs))]
                _labels, _strs, batch_tokens = self.esm1v_batch_converter(data)
                batch_tokens = batch_tokens.to(device=self.device, non_blocking=True)
                emb = self.esm1v_model(batch_tokens, repr_layers=[33], return_contacts=False)["representations"][33]
                emb = emb.transpose(2, 1)

                maxlen = emb.shape[-1]
                masks = [[1] * len(seqs[i]) + [0] * (maxlen - len(seqs[i])) for i in range(len(seqs))]
                masks = torch.tensor(masks, dtype=torch.int32, device=self.device)

                out = self.rlat_model(emb, masks)
                rlat_pred, rlat_emb, rlat_attn = [item.detach().cpu().numpy() for item in out]

                emb_pool = emb.detach().cpu().numpy().mean(axis=-1)
                emb_pool = (emb_pool - self.svr_stats[:, 0]) / (self.svr_stats[:, 1] + 1e-8)
                svr_pred = self.svr_model.predict(emb_pool)
                ensemble_pred = (rlat_pred + svr_pred) / 2
                return {
                    "rlat_pred": rlat_pred,
                    "rlat_emb": rlat_emb,
                    "rlat_attn": rlat_attn,
                    "svr_pred": svr_pred,
                    "ensemble_pred": ensemble_pred,
                }

        self.model = _EpHodModel(project_root.resolve())

    def predict_many(self, sequences: Mapping[str, str]) -> Dict[str, Optional[float]]:
        accessions = list(sequences.keys())
        seqs = [str(sequences[a]).strip().upper() for a in accessions]

        out: Dict[str, Optional[float]] = {a: None for a in accessions}
        valid: List[tuple[str, str]] = []
        for acc, seq in zip(accessions, seqs):
            if not seq:
                continue
            valid.append((acc, seq[:1022]))

        if not valid:
            return out

        # EpHod: already batch_size=1 for reproducibility; no change needed.
        for acc, seq in valid:
            pred = self.model.predict([acc], [seq])
            values = pred.get("ensemble_pred", [])
            if len(values):
                out[acc] = float(values[0])

        return out


class PatchETPretrainedPredictor(BasePredictor):
    """Predict with the pretrained PatchET model weights under patchet_pretrain_weight.

    This path does not use models.patchet.Model (which expects patch_dim in config).
    It directly loads PatchETPretrained, whose config schema matches
    patchet_pretrain_weight/model_config.yaml.
    """

    def __init__(
        self,
        model_config_path: Path,
        weight_path: Path,
        project_root: Path,
        batch_size: int = 4,  # FIX: accept batch_size
    ) -> None:
        try:
            import torch
            import yaml
            from transformers import EsmTokenizer
        except Exception as exc:
            raise ImportError("PatchET predictor requires torch, pyyaml, and transformers.") from exc

        repo_root = project_root.resolve()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        try:
            from models.patchet_pretrained import PatchETPretrained
            from utils import load_weight
        except Exception as exc:
            raise ImportError("Failed to import PatchETPretrained or load_weight from project root.") from exc

        if not model_config_path.exists():
            raise FileNotFoundError(f"PatchET config not found: {model_config_path}")
        if not weight_path.exists():
            raise FileNotFoundError(f"PatchET weight not found: {weight_path}")
        config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
        pretrain_model = str(config.get("pretrain_model", "")).strip()
        if pretrain_model:
            p = Path(pretrain_model)
            if not p.is_absolute() and (repo_root / p).exists():
                config["pretrain_model"] = str((repo_root / p).resolve())
            elif pretrain_model == "esm150":
                config["pretrain_model"] = "facebook/esm2_t30_150M_UR50D"

        self.model = PatchETPretrained(config)
        load_weight(self.model, str(weight_path))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.inference = True
        self.model.eval()

        self._torch = torch
        self.tokenizer = EsmTokenizer.from_pretrained(config["pretrain_model"])
        self.batch_size = int(batch_size)  # FIX: store batch_size

    def predict_many(self, sequences: Mapping[str, str]) -> Dict[str, Optional[float]]:
        ids = list(sequences.keys())
        seqs = [str(sequences[k]).strip().upper() for k in ids]
        out: Dict[str, Optional[float]] = {k: None for k in ids}

        valid = [(acc, seq) for acc, seq in zip(ids, seqs) if len(seq) > 0]
        if not valid:
            return out

        # FIX: process in chunks; pad only to local batch maximum, not global max_length
        for start in range(0, len(valid), self.batch_size):
            chunk = valid[start : start + self.batch_size]
            chunk_seqs = [seq for _, seq in chunk]

            inputs = self.tokenizer(
                chunk_seqs,
                return_tensors="pt",
                padding=True,       # FIX: pad to longest in THIS chunk only
                truncation=True,
                max_length=1000,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self._torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=None,
                )

            preds = outputs.pred
            for i, (acc, _seq) in enumerate(chunk):
                pred = preds[i]
                out[acc] = float(pred.item() if hasattr(pred, "item") else pred)

        return out


def _default_project_root() -> Path:
    # ped_eval/ -> PED-Eval/ -> repository root
    return Path(__file__).resolve().parents[2]


def build_predictor(
    task: str,
    predictor_name: str,
    weights_dir: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
    batch_size: int = 4,
) -> BasePredictor:
    validate_predictor_for_task(task, predictor_name)
    repo_root = Path(project_root).resolve() if project_root else _default_project_root()
    default_weights_root = (repo_root / "predictor_weights").resolve()

    if weights_dir is not None:
        weights_root = Path(weights_dir).resolve()
    elif project_root is not None:
        weights_root = default_weights_root if default_weights_root.exists() else repo_root
    else:
        weights_root = Path("predictor_weights")
        if not weights_root.exists():
            fallback_repo_root = _default_project_root()
            fallback_predictor_root = fallback_repo_root / "predictor_weights"
            weights_root = fallback_predictor_root if fallback_predictor_root.exists() else fallback_repo_root
        weights_root = weights_root.resolve()

    if predictor_name == "Seq2Topt":
        model_path = _first_existing_path(
            weights_root / "ephod" / "model_topt_window.3_r2.0.57.pth",
        )
        return Seq2RegressorPredictor(model_path=model_path, value_scale=120.0, batch_size=batch_size)

    if predictor_name == "Seq2pHopt":
        model_path = _first_existing_path(
            weights_root / "ephod" / "model_pHopt_window.3_r2.0.42.pth",
        )
        return Seq2RegressorPredictor(model_path=model_path, value_scale=14.0, batch_size=batch_size)

    if predictor_name == "PatchEX":
        model_config_path = _first_existing_path(
            weights_root / "patchex_weight" / str(task).lower() / "model_config.yaml",
        )
        weight_path = _first_existing_path(
            weights_root / "patchex_weight" / str(task).lower() / "model.safetensors",
        )
        # FIX: pass batch_size through
        return PatchInferencePredictor(
            model_config_path=model_config_path,
            weight_path=weight_path,
            project_root=repo_root,
            batch_size=batch_size,
        )

    if predictor_name == "PatchET":
        model_config_path = _first_existing_path(
            weights_root / "patchet_pretrain_weight" / "model_config.yaml",
        )
        weight_path = _first_existing_path(
            weights_root / "patchet_pretrain_weight" / "model.safetensors",
        )
        # FIX: pass batch_size through
        return PatchETPretrainedPredictor(
            model_config_path=model_config_path,
            weight_path=weight_path,
            project_root=repo_root,
            batch_size=batch_size,
        )

    if predictor_name == "EpHod":
        return EpHodPredictor(project_root=weights_root)

    raise ValueError(f"Unsupported predictor: {predictor_name}")
