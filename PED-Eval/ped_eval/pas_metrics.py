from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import pandas as pd


SIGMA_BY_TASK = {
    "opt": 4.0,
    "ph": 0.3,
}


def default_sigma(task: str) -> float:
    t = str(task).lower()
    if t not in SIGMA_BY_TASK:
        raise ValueError(f"Unsupported task for PAS sigma: {task}")
    return SIGMA_BY_TASK[t]


def compute_pas(pred_value: float, target_value: float, sigma: float) -> float:
    delta = float(pred_value) - float(target_value)
    return float(math.exp(-(delta ** 2) / (2.0 * (float(sigma) ** 2))))


def compute_pas_row(
    pred_value: Optional[float],
    target_value: Optional[float],
    sigma: float,
) -> Dict[str, Optional[float]]:
    # Legacy PAS workflow (plot/plot_PAS.py): PAS is rounded to 2 decimals,
    # and constraint satisfaction is defined by PAS > 0.
    if pred_value is None or target_value is None:
        return {
            "predicted_property": None,
            "target_value": target_value,
            "absolute_error": None,
            "PAS": None,
            "constraint_satisfied": None,
        }

    pred = float(pred_value)
    target = float(target_value)
    abs_err = abs(pred - target)
    pas_value = round(compute_pas(pred, target, sigma=sigma), 2)
    return {
        "predicted_property": pred,
        "target_value": target,
        "absolute_error": float(abs_err),
        "PAS": pas_value,
        "constraint_satisfied": bool(pas_value > 0.0),
    }


def pearson_corr(values_x: Iterable[float], values_y: Iterable[float]) -> Optional[float]:
    df = pd.DataFrame({"x": list(values_x), "y": list(values_y)}).dropna()
    if len(df) < 2:
        return None
    corr = df["x"].corr(df["y"], method="pearson")
    if pd.isna(corr):
        return None
    return float(corr)

