from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .io import (
    load_fasta_sequences,
    load_functional_annotations,
    load_native_sequence_from_pdb,
    load_task_records,
    normalize_sequence,
    split_positions_by_role,
)
from .metrics import (
    BLOSUM_MATRIX_ORDER,
    add_nan_explanations,
    compute_nssr_all_blosum,
    compute_nssr_blosum,
    compute_recovery_for_positions,
    compute_recovery_rate,
    recovery_by_secondary_structure,
)
from .pas_metrics import compute_pas_row, default_sigma, pearson_corr
from .predictors import build_predictor, validate_predictor_for_task
from .thermo_metrics import compute_ivywrel, summarize_ivywrel_analysis


@dataclass
class EvaluationReport:
    task: str
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]


_ESMFOLD_EVALUATOR_CACHE: Dict[tuple[str, str], Any] = {}
REPO_ROOT = Path(__file__).resolve().parents[2]


class _EvaluationProgressBar:
    def __init__(self, total: int, *, stream=None) -> None:
        self.total = max(total, 0)
        self.stream = stream or sys.stderr
        self.width = 32
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._last_line = ""

    def update(self, completed: int, accession: str) -> None:
        line = self._render(completed, accession)
        if line == self._last_line:
            return
        self._last_line = line
        if self.is_tty:
            self.stream.write(f"\r{line}")
        else:
            self.stream.write(f"{line}\n")
        self.stream.flush()

    def close(self) -> None:
        if self.is_tty and self.total > 0:
            self.stream.write("\n")
            self.stream.flush()

    def _render(self, completed: int, accession: str) -> str:
        total = max(self.total, 1)
        bounded_completed = min(max(completed, 0), total)
        percent = int((bounded_completed / total) * 100)
        filled = min(self.width, int(self.width * bounded_completed / total))
        bar = "#" * filled + "-" * (self.width - filled)
        return (
            f"[PED-Eval] Progress: [{bar}] {bounded_completed}/{self.total} "
            f"({percent:3d}%) current={accession}"
        )


def _resolve_runtime_path(path_value: Optional[Union[str, Path]], *, repo_root: Path = REPO_ROOT) -> Optional[Path]:
    if path_value is None:
        return None

    raw_path = Path(path_value).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    cwd_candidate = (Path.cwd() / raw_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (repo_root / raw_path).resolve()


def _default_data_path(repo_root: Path = REPO_ROOT) -> Path:
    cwd_candidate = (Path.cwd() / "data_design").resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (repo_root / "data_design").resolve()


def _default_predictor_weights_root(repo_root: Path = REPO_ROOT) -> Path:
    cwd_candidate = (Path.cwd() / "predictor_weights").resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (repo_root / "predictor_weights").resolve()


def _is_within_repo_subdir(path_value: Optional[Union[str, Path]], subdir_name: str, *, repo_root: Path = REPO_ROOT) -> bool:
    resolved_path = _resolve_runtime_path(path_value, repo_root=repo_root)
    if resolved_path is None:
        return False

    try:
        relative_path = resolved_path.relative_to(repo_root)
    except ValueError:
        return False
    return bool(relative_path.parts) and relative_path.parts[0] == subdir_name


def _load_setup_assets_api(repo_root: Path = REPO_ROOT):
    try:
        from setup_assets import AssetSetupError, build_asset_targets, ensure_assets

        return ensure_assets, AssetSetupError, build_asset_targets
    except (ImportError, SyntaxError):
        setup_assets_path = repo_root / "setup_assets.py"
        if not setup_assets_path.exists():
            return None

        spec = importlib.util.spec_from_file_location("patchex_setup_assets", setup_assets_path)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("patchex_setup_assets", module)
        try:
            spec.loader.exec_module(module)
        except (ImportError, SyntaxError):
            return None
        return module.ensure_assets, module.AssetSetupError, module.build_asset_targets


def _ensure_repo_assets_for_evaluation(
    *,
    dataset_csv: Path,
    pdb_dir: Path,
    functional_sites_long_csv: Path,
    enable_pas: bool,
    predictor_weights_dir: Optional[Union[str, Path]],
    predictor_project_root: Optional[Union[str, Path]],
    auto_setup_assets: bool,
    repo_root: Path = REPO_ROOT,
) -> None:
    if not auto_setup_assets:
        return

    setup_assets_api = _load_setup_assets_api(repo_root)
    if setup_assets_api is None:
        return

    ensure_assets, AssetSetupError, build_asset_targets = setup_assets_api
    asset_targets = build_asset_targets(repo_root)
    requested_bundles: list[str] = []

    uses_repo_data = any(
        _is_within_repo_subdir(path, "data_design", repo_root=repo_root)
        for path in (dataset_csv, pdb_dir, functional_sites_long_csv)
    )
    if uses_repo_data and not asset_targets["data_design"].is_installed():
        requested_bundles.append("ped-eval")

    predictor_weights_root: Optional[Path] = None
    if enable_pas:
        if predictor_weights_dir is not None:
            predictor_weights_root = _resolve_runtime_path(predictor_weights_dir, repo_root=repo_root)
        elif predictor_project_root is not None:
            predictor_project_root_path = _resolve_runtime_path(predictor_project_root, repo_root=repo_root)
            if predictor_project_root_path is not None:
                predictor_weights_root = (predictor_project_root_path / "predictor_weights").resolve()
        else:
            predictor_weights_root = _default_predictor_weights_root(repo_root)

    if (
        predictor_weights_root is not None
        and _is_within_repo_subdir(predictor_weights_root, "predictor_weights", repo_root=repo_root)
        and not asset_targets["predictor_weights"].is_installed()
        and "ped-eval" not in requested_bundles
    ):
        requested_bundles.append("ped-eval")

    if not requested_bundles:
        return

    try:
        ensure_assets(bundles=tuple(requested_bundles), repo_root=repo_root)
    except AssetSetupError as exc:
        raise RuntimeError(f"Automatic PED-Eval asset setup failed: {exc}") from exc


def _default_dataset_csv(task: str, data_path: Optional[Path] = None, repo_root: Path = REPO_ROOT) -> Path:
    base = data_path or _default_data_path(repo_root)
    return base / f"{task}.csv"


def _default_pdb_dir(data_path: Optional[Path] = None, repo_root: Path = REPO_ROOT) -> Path:
    base = data_path or _default_data_path(repo_root)
    return base / "pdb"


def _default_sites_long_csv(task: str, data_path: Optional[Path] = None, repo_root: Path = REPO_ROOT) -> Path:
    base = data_path or _default_data_path(repo_root)
    return base / "functional_site_annotations" / task / "functional_sites_long.csv"


def _flatten_for_csv(result: Dict[str, Any]) -> Dict[str, Any]:
    flat = {
        "accession": result.get("accession"),
        "status": result.get("status", "ok"),
        "error": result.get("error", ""),
        "sequence_length": result.get("sequence_length"),
        "n_functional_sites": result.get("n_functional_sites"),
        "n_catalytic_sites": result.get("n_catalytic_sites"),
        "n_binding_sites": result.get("n_binding_sites"),
        "recovery_rate": result.get("recovery_rate"),
        "NSSR": result.get("NSSR"),
    }

    # BLOSUM breakdown
    for k in BLOSUM_MATRIX_ORDER:
        key = f"NSSR_BLOSUM{k}"
        flat[key] = result.get(key)

    flat["E-FSR"] = result.get("E-FSR")
    flat["C-FSR"] = result.get("C-FSR")

    cat = result.get("catalytic_site_recovery", {}) if isinstance(result.get("catalytic_site_recovery"), dict) else {}
    bind = result.get("binding_site_recovery", {}) if isinstance(result.get("binding_site_recovery"), dict) else {}
    flat["catalytic_E-FSR"] = cat.get("E-FSR")
    flat["catalytic_C-FSR"] = cat.get("C-FSR")
    flat["binding_E-FSR"] = bind.get("E-FSR")
    flat["binding_C-FSR"] = bind.get("C-FSR")

    ss = result.get("recovery_by_secondary_structure", {})
    if isinstance(ss, dict):
        for group, stats in ss.items():
            if not isinstance(stats, dict):
                continue
            safe_group = group.replace("-", "_").replace(" ", "_")
            flat[f"ss_{safe_group}_recovery_rate"] = stats.get("recovery_rate")

    # Optional structure metrics
    for m in ("mean_plddt", "tm_score", "rmsd"):
        if m in result:
            flat[m] = result[m]

    # Optional PAS metrics
    for m in ("target_value", "predicted_property", "absolute_error", "PAS", "constraint_satisfied"):
        if m in result:
            flat[m] = result[m]

    # Optional thermophilic composition proxies
    for m in ("IVYWREL",):
        if m in result:
            flat[m] = result[m]

    return flat


def _build_summary(task: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [r for r in results if r.get("status") == "ok"]
    err_rows = [r for r in results if r.get("status") != "ok"]

    summary: Dict[str, Any] = {
        "task": task,
        "n_total": len(results),
        "n_success": len(ok_rows),
        "n_error": len(err_rows),
        "error_examples": [
            {"accession": r.get("accession"), "error": r.get("error")} for r in err_rows[:10]
        ],
    }

    if ok_rows:
        df = pd.DataFrame([_flatten_for_csv(r) for r in ok_rows])

        scalar_keys = (
            ["recovery_rate", "NSSR"]
            + [f"NSSR_BLOSUM{k}" for k in BLOSUM_MATRIX_ORDER]
            + ["E-FSR", "C-FSR", "catalytic_E-FSR", "catalytic_C-FSR", "binding_E-FSR", "binding_C-FSR"]
        )
        for key in scalar_keys:
            if key in df.columns:
                vals = df[key].dropna()
                summary[f"mean_{key}"] = float(vals.mean()) if len(vals) else None

        # Secondary structure breakdown in summary
        ss_cols = [c for c in df.columns if c.startswith("ss_") and c.endswith("_recovery_rate")]
        if ss_cols:
            ss_summary: Dict[str, Optional[float]] = {}
            for col in ss_cols:
                vals = df[col].dropna()
                ss_summary[col] = float(vals.mean()) if len(vals) else None
            summary["mean_secondary_structure_recovery"] = ss_summary

        # Optional structure metrics
        for m in ("mean_plddt", "tm_score", "rmsd"):
            if m in df.columns:
                vals = df[m].dropna()
                summary[f"mean_{m}"] = float(vals.mean()) if len(vals) else None

        # Optional PAS metrics
        if "PAS" in df.columns:
            vals = df["PAS"].dropna()
            summary["mean_PAS"] = float(vals.mean()) if len(vals) else None
        if "constraint_satisfied" in df.columns:
            vals = df["constraint_satisfied"].dropna()
            summary["constraint_satisfaction_rate"] = float(vals.mean()) if len(vals) else None
        if "absolute_error" in df.columns:
            vals = df["absolute_error"].dropna()
            summary["MAE"] = float(vals.mean()) if len(vals) else None
        if "predicted_property" in df.columns and "target_value" in df.columns:
            summary["Pearson_correlation"] = pearson_corr(df["predicted_property"], df["target_value"])

        # Optional thermophilic composition proxies
        if "IVYWREL" in df.columns:
            vals = pd.to_numeric(df["IVYWREL"], errors="coerce").dropna()
            summary["mean_IVYWREL"] = float(vals.mean()) if len(vals) else None
            summary["std_IVYWREL"] = float(vals.std(ddof=1)) if len(vals) >= 2 else None
            summary["n_IVYWREL"] = int(len(vals))
            if task == "opt" and "target_value" in df.columns:
                summary["IVYWREL_analysis"] = summarize_ivywrel_analysis(
                    df,
                    ivywrel_col="IVYWREL",
                    target_col="target_value",
                )

    return summary


def evaluate_single_sequence(
    accession: str,
    predicted_sequence: str,
    native_sequence: str,
    reference_pdb: Optional[Union[str, Path]] = None,
    use_esmfold: bool = False,
    esmfold_output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Evaluate one designed sequence.

    ``recovery_rate`` is the canonical PED-Eval field and ``sequence_recovery``
    is the same value exposed for compatibility with older pipeline outputs.
    """
    accession = str(accession)
    native_sequence = normalize_sequence(native_sequence)
    predicted_sequence = normalize_sequence(predicted_sequence)

    if len(native_sequence) != len(predicted_sequence):
        raise ValueError(
            f"Sequence length mismatch for {accession}: native sequence length {len(native_sequence)}, "
            f"predicted sequence length {len(predicted_sequence)}. "
            "Please ensure both sequences represent the same protein and are properly aligned."
        )

    sequence_recovery = compute_recovery_rate(native_sequence, predicted_sequence)
    result: Dict[str, Any] = {
        "accession": accession,
        "sequence_length": len(native_sequence),
        "recovery_rate": sequence_recovery,
        "sequence_recovery": sequence_recovery,
    }

    if use_esmfold:
        if reference_pdb is None:
            raise ValueError("reference_pdb is required when use_esmfold=True")
        from .structure import ESMFoldEvaluator

        reference_pdb = Path(reference_pdb)
        fold_out = Path(esmfold_output_dir) if esmfold_output_dir else Path("esmfold_output")
        fold_out.mkdir(parents=True, exist_ok=True)
        cache_key = (str(reference_pdb.parent.resolve()), str(fold_out.resolve()))
        esmfold_evaluator = _ESMFOLD_EVALUATOR_CACHE.get(cache_key)
        if esmfold_evaluator is None:
            esmfold_evaluator = ESMFoldEvaluator(pdb_dir=reference_pdb.parent, output_dir=fold_out)
            _ESMFOLD_EVALUATOR_CACHE[cache_key] = esmfold_evaluator
        result.update(
            esmfold_evaluator(
                accession=accession,
                sequence=predicted_sequence,
                reference_pdb=reference_pdb,
            )
        )

    return result


def evaluate_task(
    task: str,
    result_fasta: Union[str, Path],
    dataset_csv: Optional[Union[str, Path]] = None,
    pdb_dir: Optional[Union[str, Path]] = None,
    functional_sites_long_csv: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
    use_esmfold: bool = False,
    esmfold_output_dir: Optional[Union[str, Path]] = None,
    enable_pas: bool = False,
    enable_thermo_proxies: bool = False,
    predictor: Optional[str] = None,
    pas_sigma: Optional[float] = None,
    predictor_weights_dir: Optional[Union[str, Path]] = None,
    predictor_project_root: Optional[Union[str, Path]] = None,
    predictor_batch_size: int = 4,
    auto_setup_assets: bool = True,
    show_progress: bool = False,
) -> EvaluationReport:
    task = str(task).lower()
    if task not in {"opt", "ph"}:
        raise ValueError("task must be 'opt' or 'ph'")
    if enable_thermo_proxies and task != "opt":
        raise ValueError("Thermophilic composition proxies are only supported for task='opt'.")

    result_fasta = _resolve_runtime_path(result_fasta, repo_root=REPO_ROOT)
    _data_path = _resolve_runtime_path(data_path, repo_root=REPO_ROOT) if data_path else None
    dataset_csv = _resolve_runtime_path(dataset_csv, repo_root=REPO_ROOT) if dataset_csv else _default_dataset_csv(task, _data_path, repo_root=REPO_ROOT)
    pdb_dir = _resolve_runtime_path(pdb_dir, repo_root=REPO_ROOT) if pdb_dir else _default_pdb_dir(_data_path, repo_root=REPO_ROOT)
    functional_sites_long_csv = _resolve_runtime_path(functional_sites_long_csv, repo_root=REPO_ROOT) if functional_sites_long_csv else _default_sites_long_csv(task, _data_path, repo_root=REPO_ROOT)

    _ensure_repo_assets_for_evaluation(
        dataset_csv=dataset_csv,
        pdb_dir=pdb_dir,
        functional_sites_long_csv=functional_sites_long_csv,
        enable_pas=bool(enable_pas or predictor),
        predictor_weights_dir=predictor_weights_dir,
        predictor_project_root=predictor_project_root,
        auto_setup_assets=auto_setup_assets,
        repo_root=REPO_ROOT,
    )

    records = load_task_records(dataset_csv, task=task)
    predicted = load_fasta_sequences(result_fasta)
    sites_by_accession, role_map = load_functional_annotations(functional_sites_long_csv)

    # Optional PAS predictor
    pas_enabled = bool(enable_pas or predictor)
    predictor_name = predictor
    if pas_enabled:
        if not predictor_name:
            raise ValueError("PAS metrics requested but no predictor was provided. Use --predictor.")
        validate_predictor_for_task(task, predictor_name)
        sigma = float(pas_sigma) if pas_sigma is not None else default_sigma(task)
        property_predictor = build_predictor(
            task=task,
            predictor_name=predictor_name,
            weights_dir=predictor_weights_dir,
            project_root=predictor_project_root,
            batch_size=int(predictor_batch_size),
        )
        pred_property_by_accession = property_predictor.predict_many(predicted)
    else:
        sigma = None
        pred_property_by_accession = {}

    # Optionally load ESM-fold evaluator
    esmfold_evaluator = None
    if use_esmfold:
        from .structure import ESMFoldEvaluator
        fold_out = Path(esmfold_output_dir) if esmfold_output_dir else Path("esmfold_output")
        fold_out.mkdir(parents=True, exist_ok=True)
        esmfold_evaluator = ESMFoldEvaluator(pdb_dir=pdb_dir, output_dir=fold_out)

    results: List[Dict[str, Any]] = []
    progress_bar = _EvaluationProgressBar(len(predicted)) if show_progress and len(predicted) > 0 else None
    completed = 0

    for accession, predicted_sequence in predicted.items():
        try:
            if accession not in records:
                raise KeyError(f"Accession not found in dataset: {accession}")

            record = records[accession]
            native_sequence = load_native_sequence_from_pdb(pdb_dir / f"{accession}.pdb")
            predicted_sequence = normalize_sequence(predicted_sequence)
            sec_structure = str(record.sec_structure)

            if len(native_sequence) != len(predicted_sequence):
                raise ValueError(
                    f"Predicted sequence length mismatch for {accession}: expected {len(native_sequence)}, got {len(predicted_sequence)}"
                )
            if len(native_sequence) != len(sec_structure):
                raise ValueError(
                    f"Secondary structure length mismatch for {accession}: sequence={len(native_sequence)}, sec_structure={len(sec_structure)}"
                )

            site_positions = sites_by_accession.get(accession, set())
            site_roles = split_positions_by_role(accession, site_positions, role_map)

            efsr, cfsr, n_func_req, n_func_valid = compute_recovery_for_positions(
                native_sequence, predicted_sequence, site_positions, strict_key_chem=False
            )
            cat_e, cat_c, n_cat_req, n_cat_valid = compute_recovery_for_positions(
                native_sequence, predicted_sequence, site_roles["catalytic"], strict_key_chem=True
            )
            bind_e, bind_c, n_bind_req, n_bind_valid = compute_recovery_for_positions(
                native_sequence, predicted_sequence, site_roles["binding"], strict_key_chem=False
            )

            nssr_all = compute_nssr_all_blosum(native_sequence, predicted_sequence)

            result: Dict[str, Any] = {
                "accession": accession,
                "status": "ok",
                "sequence_length": len(native_sequence),
                "n_functional_sites": len(site_positions),
                "n_catalytic_sites": len(site_roles["catalytic"]),
                "n_binding_sites": len(site_roles["binding"]),
                "recovery_rate": compute_recovery_rate(native_sequence, predicted_sequence),
                "NSSR": compute_nssr_blosum(native_sequence, predicted_sequence),
                **nssr_all,
                "recovery_by_secondary_structure": recovery_by_secondary_structure(native_sequence, predicted_sequence, sec_structure),
                "E-FSR": efsr,
                "C-FSR": cfsr,
                "catalytic_site_recovery": {"E-FSR": cat_e, "C-FSR": cat_c},
                "binding_site_recovery": {"E-FSR": bind_e, "C-FSR": bind_c},
            }

            if enable_thermo_proxies:
                result["target_value"] = record.target_value

            if pas_enabled:
                pas_fields = compute_pas_row(
                    pred_value=pred_property_by_accession.get(accession),
                    target_value=record.target_value,
                    sigma=float(sigma),
                )
                result.update(pas_fields)

            if enable_thermo_proxies:
                result["IVYWREL"] = compute_ivywrel(predicted_sequence)

            add_nan_explanations(
                result,
                diagnostics={
                    "functional": {"requested": n_func_req, "valid": n_func_valid},
                    "catalytic": {"requested": n_cat_req, "valid": n_cat_valid},
                    "binding": {"requested": n_bind_req, "valid": n_bind_valid},
                },
            )

            # Optional ESM-fold structure metrics
            if esmfold_evaluator is not None:
                struct_metrics = esmfold_evaluator(accession=accession, sequence=predicted_sequence)
                result.update(struct_metrics)

            results.append(result)

        except Exception as exc:
            results.append({
                "accession": accession,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
        finally:
            completed += 1
            if progress_bar is not None:
                progress_bar.update(completed, accession)

    if progress_bar is not None:
        progress_bar.close()

    summary = _build_summary(task, results)
    if pas_enabled:
        summary["pas_predictor"] = predictor_name
        summary["pas_sigma"] = sigma
        summary["constraint_definition"] = "round(PAS, 2) > 0"
    if enable_thermo_proxies:
        summary["thermo_proxy_metrics"] = ["IVYWREL"]

    return EvaluationReport(task=task, results=results, summary=summary)


def write_report(report: EvaluationReport, output_dir: Union[str, Path]) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "per_accession.jsonl"
    csv_path = output_dir / "per_accession.csv"
    summary_path = output_dir / "summary.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in report.results:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    pd.DataFrame([_flatten_for_csv(r) for r in report.results]).to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(report.summary, indent=2, ensure_ascii=True), encoding="utf-8")

    return {"jsonl": jsonl_path, "csv": csv_path, "summary": summary_path}

