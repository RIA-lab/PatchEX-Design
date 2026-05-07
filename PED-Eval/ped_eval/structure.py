"""Optional ESM-fold based structure evaluation.

This module is only imported when ``--use-esmfold`` is passed on the CLI.
It requires ``transformers`` and a GPU is strongly recommended.
"""
from __future__ import annotations

import subprocess
import re
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# TM-align helper (reused from the project's evaluation/structure.py)
# ---------------------------------------------------------------------------

def _compute_tmscore(pdb1: str, pdb2: str) -> Tuple[Optional[float], Optional[float]]:
    """Run TM-align and return (tm_score, rmsd). Returns (None, None) on failure."""
    for p in (pdb1, pdb2):
        if not os.path.exists(p):
            return None, None
    try:
        result = subprocess.run(
            ["TMalign", pdb1, pdb2],
            capture_output=True, text=True, check=True, timeout=120,
        )
        out = result.stdout
        tm_match = re.search(r"TM-score=\s*([\d\.]+)", out)
        rmsd_match = re.search(r"RMSD=\s*([\d\.]+)", out)
        tm_score = float(tm_match.group(1)) if tm_match else None
        rmsd = float(rmsd_match.group(1)) if rmsd_match else None
        return tm_score, rmsd
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# PDB writer (backbone only, compatible with TM-align)
# ---------------------------------------------------------------------------

_AA1_TO_3 = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
    'X': 'UNK',
}

_BACKBONE = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]


def _write_pdb(sequence: str, coords, plddt, out_path: str) -> None:
    with open(out_path, "w") as f:
        atom_index = 1
        for res_i, (aa, atom_coords, b) in enumerate(zip(sequence, coords, plddt), start=1):
            aa3 = _AA1_TO_3.get(aa.upper(), "UNK")
            for atom_j in range(min(4, len(atom_coords))):
                atom_name, element = _BACKBONE[atom_j]
                x, y, z = atom_coords[atom_j]
                f.write(
                    f"ATOM  {atom_index:5d}  {atom_name:<3s} {aa3:>3s} A{res_i:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           {element:>1s}\n"
                )
                atom_index += 1
        f.write("END\n")


# ---------------------------------------------------------------------------
# ESMFoldEvaluator
# ---------------------------------------------------------------------------

class ESMFoldEvaluator:
    """Fold a predicted sequence with ESMFold, then compute structural metrics.

    Parameters
    ----------
    pdb_dir:
        Directory containing native ``.pdb`` files named ``<accession>.pdb``.
    output_dir:
        Where to write predicted ``<accession>_pred.pdb`` files.
    """

    def __init__(self, pdb_dir: str | Path, output_dir: str | Path) -> None:
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding

        self.pdb_dir = Path(pdb_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device)
        self.model.eval()

    def __call__(self, accession: str, sequence: str) -> Dict[str, Any]:
        import torch

        inputs = self.tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        coords = outputs.positions[-1, 0].detach().cpu().numpy()
        plddt_per_res = outputs.plddt[0].mean(dim=-1).detach().cpu().numpy()

        del outputs, inputs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        pred_pdb = str(self.output_dir / f"{accession}_pred.pdb")
        _write_pdb(sequence, coords, plddt_per_res, pred_pdb)

        mean_plddt = float(plddt_per_res.mean())

        ref_pdb = str(self.pdb_dir / f"{accession}.pdb")
        tm_score, rmsd = _compute_tmscore(pred_pdb, ref_pdb)

        return {
            "mean_plddt": mean_plddt,
            "tm_score": tm_score,
            "rmsd": rmsd,
        }

