from __future__ import annotations

from typing import Any
from typing import AbstractSet

import pandas as pd


IVYWREL_RESIDUES = frozenset("IVYWREL")
DEFAULT_TEMPERATURE_BINS = [0.0, 20.0, 37.0, 50.0, 65.0, 80.0, float("inf")]
DEFAULT_TEMPERATURE_BIN_LABELS = ["<20°C", "20-37°C", "37-50°C", "50-65°C", "65-80°C", ">80°C"]


def _normalize_sequence(sequence: str) -> str:
    return "".join(ch for ch in str(sequence).strip().upper() if ch.isalpha())


def compute_residue_fraction(sequence: str, residues: AbstractSet[str]) -> float:
    """Return the fraction of residues in ``sequence`` that belong to ``residues``."""
    seq = _normalize_sequence(sequence)
    if not seq:
        return float("nan")
    return sum(aa in residues for aa in seq) / len(seq)


def compute_ivywrel(sequence: str) -> float:
    """Thermophilic composition proxy based on the IVYWREL residue set."""
    return compute_residue_fraction(sequence, IVYWREL_RESIDUES)


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def summarize_ivywrel_analysis(
    df: pd.DataFrame,
    ivywrel_col: str = "IVYWREL",
    target_col: str = "target_value",
) -> dict:
    """Build no-plot IVYWREL analysis metadata for summary.json."""
    ivywrel = pd.to_numeric(df.get(ivywrel_col), errors="coerce")
    targets = pd.to_numeric(df.get(target_col), errors="coerce")

    valid = pd.DataFrame({"IVYWREL": ivywrel, "target_value": targets}).dropna()

    if len(valid) >= 2:
        pearson_r = _safe_float(valid["IVYWREL"].corr(valid["target_value"], method="pearson"))
        spearman_r = _safe_float(valid["IVYWREL"].corr(valid["target_value"], method="spearman"))
    else:
        pearson_r = None
        spearman_r = None

    binned = valid.copy()
    if len(binned):
        binned["target_bin"] = pd.cut(
            binned["target_value"],
            bins=DEFAULT_TEMPERATURE_BINS,
            labels=DEFAULT_TEMPERATURE_BIN_LABELS,
            right=False,
        )
        grouped = binned.groupby("target_bin", observed=False)["IVYWREL"].agg(["mean", "std", "count"])
    else:
        grouped = pd.DataFrame(index=DEFAULT_TEMPERATURE_BIN_LABELS, columns=["mean", "std", "count"])

    by_target_bin = []
    for label in DEFAULT_TEMPERATURE_BIN_LABELS:
        if label in grouped.index:
            row = grouped.loc[label]
            count = int(0 if pd.isna(row.get("count")) else row["count"])
            mean = _safe_float(row.get("mean"))
            std = _safe_float(row.get("std"))
        else:
            count = 0
            mean = None
            std = None
        by_target_bin.append(
            {
                "label": label,
                "mean_IVYWREL": mean,
                "std_IVYWREL": std,
                "count": count,
            }
        )

    return {
        "target_correlation": {
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
            "n_samples": int(len(valid)),
        },
        "by_target_bin": by_target_bin,
    }


