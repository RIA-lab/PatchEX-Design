#!/usr/bin/env python3
"""
Condition alignment metrics for enzyme design.

Metrics implemented:
  1. Conditional Controllability — does predicted property change as target condition changes?
  2. Mutual Alignment (ΔpH tolerance) — how close are predicted vs. desired values?
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# ===========================================
# 1️⃣ Conditional Controllability
# ===========================================
def conditional_controllability(predicted_values, target_values):
    """
    Measure how well predicted property changes follow the direction and magnitude
    of target condition changes. Essentially the correlation between deltas.

    Args:
        predicted_values (list or np.ndarray): Predicted property values (e.g. predicted pH).
        target_values (list or np.ndarray): Target condition values (e.g. desired pH).

    Returns:
        float: Controllability score (0–1), higher means better alignment.
    """
    predicted_values = np.asarray(predicted_values, dtype=float)
    target_values = np.asarray(target_values, dtype=float)

    # Require at least 3 samples to compute meaningful correlation
    if len(predicted_values) < 3:
        return np.nan

    # Compute pairwise deltas
    delta_pred = np.diff(predicted_values)
    delta_target = np.diff(target_values)

    # If all targets are identical, controllability is undefined
    if np.all(delta_target == 0):
        return np.nan

    # Compute correlation between Δtarget and Δpred
    r, _ = pearsonr(delta_target, delta_pred)

    # Normalize from [-1, 1] → [0, 1] for interpretability
    controllability = (r + 1) / 2
    return controllability


# ===========================================
# 2️⃣ Mutual Alignment (ΔpH tolerance)
# ===========================================
def mutual_alignment(predicted_values, target_values, tolerance=1.0):
    """
    Measure how closely predicted property aligns with the desired condition.

    Args:
        predicted_values (list or np.ndarray): Predicted property values (e.g. predicted pH)
        target_values (list or np.ndarray): Target condition values (e.g. desired pH)
        tolerance (float): Acceptable absolute error (e.g., ±1.0 pH unit)

    Returns:
        dict with:
            - mean_abs_error
            - within_tolerance_rate (%)
            - delta_ph_list (list of individual differences)
    """
    predicted_values = np.asarray(predicted_values, dtype=float)
    target_values = np.asarray(target_values, dtype=float)
    delta = np.abs(predicted_values - target_values)

    mae = np.mean(delta)
    within_tol = np.mean(delta <= tolerance) * 100.0

    return {
        "mean_abs_error": float(mae),
        "within_tolerance_rate": float(within_tol),
        "delta_ph_list": delta.tolist()
    }


# ===========================================
# 3️⃣ Batch evaluation utility
# ===========================================
def evaluate_condition_metrics(df, predicted_col="predicted_pH", target_col="target_pH", tolerance=1.0):
    """
    Evaluate condition alignment metrics on a DataFrame.

    Args:
        df (pd.DataFrame): Must contain columns with predicted and target values.
        predicted_col (str): Column name for predicted property.
        target_col (str): Column name for target property.
        tolerance (float): pH tolerance for "within_tolerance_rate".

    Returns:
        dict: summary of metrics
    """
    preds = df[predicted_col].values
    targets = df[target_col].values

    controllability = conditional_controllability(preds, targets)
    align_metrics = mutual_alignment(preds, targets, tolerance=tolerance)
    corr, _ = pearsonr(preds, targets)

    summary = {
        "pearson_correlation": float(corr),
        "conditional_controllability": float(controllability),
        "mean_abs_error": align_metrics["mean_abs_error"],
        "within_tolerance_rate": align_metrics["within_tolerance_rate"]
    }
    return summary


# ===========================================
# Example usage
# ===========================================
if __name__ == "__main__":
    # Example: predicted vs. target pH
    predicted = [6.5, 7.2, 7.8, 8.5, 9.1]
    target =    [6.0, 7.0, 8.0, 9.0, 10.0]

    summary = evaluate_condition_metrics(
        pd.DataFrame({"predicted_pH": predicted, "target_pH": target}),
        predicted_col="predicted_pH",
        target_col="target_pH",
        tolerance=0.5
    )

    print("\n=== Conditional Alignment Metrics ===")
    for k, v in summary.items():
        print(f"{k:30s}: {v:.4f}")
