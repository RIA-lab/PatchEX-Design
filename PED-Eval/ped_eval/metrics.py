from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Set, Tuple

from Bio.Align import substitution_matrices


BLOSUM62 = substitution_matrices.load("BLOSUM62")
BLOSUM90 = substitution_matrices.load("BLOSUM90")
BLOSUM80 = substitution_matrices.load("BLOSUM80")
BLOSUM50 = substitution_matrices.load("BLOSUM50")
BLOSUM45 = substitution_matrices.load("BLOSUM45")

BLOSUM_MATRICES = {
    90: BLOSUM90,
    80: BLOSUM80,
    62: BLOSUM62,
    50: BLOSUM50,
    45: BLOSUM45,
}
BLOSUM_MATRIX_ORDER: List[int] = [90, 80, 62, 50, 45]

DSSP_GROUPS = {
    "H": "alpha-helix",
    "G": "3_10-helix",
    "I": "pi-helix",
    "E": "beta-sheet",
    "B": "beta-bridge",
    "T": "h-bonded turn",
    "S": "bend",
    "-": "coil",
    " ": "coil",
}

KEY_CHEM = set("HDEKSCY")
FUNC_CLASS = {
    "D": "acidic",
    "E": "acidic",
    "K": "basic",
    "R": "basic",
    "H": "basic",
    "S": "polar",
    "T": "polar",
    "N": "polar",
    "Q": "polar",
    "A": "hydrophobic",
    "V": "hydrophobic",
    "I": "hydrophobic",
    "L": "hydrophobic",
    "M": "hydrophobic",
    "P": "hydrophobic",
    "G": "hydrophobic",
    "F": "aromatic",
    "Y": "aromatic",
    "W": "aromatic",
    "C": "special",
}


def blosum_score(a1: str, a2: str, matrix=BLOSUM62) -> Optional[float]:
    try:
        return matrix[a1.upper(), a2.upper()]
    except KeyError:
        return None


def compute_recovery_rate(native_seq: str, pred_seq: str) -> float:
    """Exact amino acid identity ratio (recovery_rate)."""
    if len(native_seq) != len(pred_seq):
        raise ValueError(f"Sequence length mismatch: native={len(native_seq)}, predicted={len(pred_seq)}")
    if not native_seq:
        return float("nan")
    return sum(n == p for n, p in zip(native_seq, pred_seq)) / len(native_seq)


def compute_nssr_blosum(native_seq: str, pred_seq: str, matrix=BLOSUM62) -> float:
    if len(native_seq) != len(pred_seq):
        raise ValueError(f"Sequence length mismatch for NSSR: native={len(native_seq)}, predicted={len(pred_seq)}")
    if not native_seq:
        return float("nan")
    correct = sum((blosum_score(n, p, matrix) or float("-inf")) > 0 for n, p in zip(native_seq, pred_seq))
    return correct / len(native_seq)


def compute_nssr_all_blosum(native_seq: str, pred_seq: str) -> Dict[str, float]:
    """Compute NSSR under each BLOSUM matrix."""
    return {
        f"NSSR_BLOSUM{k}": compute_nssr_blosum(native_seq, pred_seq, matrix=m)
        for k, m in BLOSUM_MATRICES.items()
    }


def recovery_by_secondary_structure(true_seq: str, pred_seq: str, ss_seq: str) -> Dict[str, Dict[str, float]]:
    if not (len(true_seq) == len(pred_seq) == len(ss_seq)):
        raise ValueError(
            f"Length mismatch for secondary-structure recovery: true={len(true_seq)}, predicted={len(pred_seq)}, sec_structure={len(ss_seq)}"
        )

    stats = defaultdict(lambda: [0, 0])
    for t, p, s in zip(true_seq, pred_seq, ss_seq):
        group = DSSP_GROUPS.get(s, "coil")
        stats[group][1] += 1
        if t == p:
            stats[group][0] += 1

    return {
        group: {"correct": correct, "total": total, "recovery_rate": (correct / total if total else float("nan"))}
        for group, (correct, total) in stats.items()
    }


def is_conservative(native_aa: str, pred_aa: str, strict_key_chem: bool, blosum_threshold: int = 0) -> bool:
    native_aa = native_aa.upper()
    pred_aa = pred_aa.upper()

    if native_aa == pred_aa:
        return True
    if strict_key_chem and native_aa in KEY_CHEM:
        return False

    score = blosum_score(native_aa, pred_aa)
    if score is not None and score >= blosum_threshold:
        return True

    return FUNC_CLASS.get(native_aa) == FUNC_CLASS.get(pred_aa)


def compute_recovery_for_positions(native_seq: str, pred_seq: str, positions: Set[int], strict_key_chem: bool) -> Tuple[float, float, int, int]:
    if not positions:
        return float("nan"), float("nan"), 0, 0

    max_len = min(len(native_seq), len(pred_seq))
    exact_ok = 0
    cons_ok = 0
    valid = 0
    requested = len(positions)

    for pos in positions:
        idx = pos - 1
        if idx < 0 or idx >= max_len:
            continue
        valid += 1
        n = native_seq[idx]
        p = pred_seq[idx]

        if n == p:
            exact_ok += 1
            cons_ok += 1
        elif is_conservative(n, p, strict_key_chem=strict_key_chem):
            cons_ok += 1

    if valid == 0:
        return float("nan"), float("nan"), requested, 0

    return exact_ok / valid, cons_ok / valid, requested, valid


def build_nan_reason(scope: str, requested_positions: int, valid_positions: int) -> Optional[Dict[str, object]]:
    if requested_positions > 0 and valid_positions > 0:
        return None

    reason: Dict[str, object] = {
        "scope": scope,
        "requested_positions": requested_positions,
        "valid_positions": valid_positions,
    }

    if requested_positions == 0:
        reason.update(
            {
                "reason": "no_annotated_positions",
                "message": f"No annotated {scope} positions were found for this accession.",
            }
        )
    else:
        reason.update(
            {
                "reason": "no_valid_positions_after_bounds_check",
                "message": f"{requested_positions} annotated {scope} positions were found, but none map inside the compared sequence length.",
            }
        )

    return reason


def add_nan_explanations(result: Dict[str, object], diagnostics: Mapping[str, Mapping[str, int]]) -> None:
    nan_explanations: Dict[str, Dict[str, object]] = {}

    nssr = result.get("NSSR")
    if isinstance(nssr, float) and math.isnan(nssr):
        nan_explanations["NSSR"] = {
            "reason": "empty_normalized_sequence",
            "message": "NSSR is undefined because the normalized native sequence is empty.",
        }

    overall_reason = build_nan_reason("functional", diagnostics["functional"]["requested"], diagnostics["functional"]["valid"])
    if overall_reason:
        for key in ("E-FSR", "C-FSR"):
            value = result.get(key)
            if isinstance(value, float) and math.isnan(value):
                nan_explanations[key] = overall_reason

    for scope, nested_key in (("catalytic", "catalytic_site_recovery"), ("binding", "binding_site_recovery")):
        nested = result.get(nested_key)
        reason = build_nan_reason(scope, diagnostics[scope]["requested"], diagnostics[scope]["valid"])
        if reason and isinstance(nested, dict):
            e_val = nested.get("E-FSR")
            if isinstance(e_val, float) and math.isnan(e_val):
                nested["explanation"] = reason
                nan_explanations[nested_key] = reason

    if nan_explanations:
        result["nan_explanations"] = nan_explanations

