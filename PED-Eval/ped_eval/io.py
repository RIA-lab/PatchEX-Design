from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Set, Tuple, Union

import pandas as pd


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "ASX": "B",
    "GLX": "Z",
    "UNK": "X",
}


@dataclass
class TaskRecord:
    accession: str
    sec_structure: str
    target_value: Optional[float] = None


def normalize_sequence(sequence: str) -> str:
    return str(sequence).strip().upper().replace("-", "")


def load_fasta_sequences(fasta_path: Union[str, Path]) -> Dict[str, str]:
    fasta_path = Path(fasta_path)
    sequences: Dict[str, str] = {}
    current_id: Optional[str] = None
    current_seq: list[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = normalize_sequence("".join(current_seq))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        sequences[current_id] = normalize_sequence("".join(current_seq))

    return sequences


def _coerce_float_or_none(value: object) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        if isinstance(value, (int, float, str)):
            return float(value)
        return float(str(value))
    except Exception:
        return None


def _detect_target_column(columns: Iterable[str], task: Optional[str]) -> Optional[str]:
    col_set = {str(c) for c in columns}
    candidates_by_task = {
        "opt": ["temperature_optimum", "target_value", "topt", "opt", "temperature"],
        "ph": ["ph", "pH", "ph_optimum", "pH_optimum", "target_value"],
    }
    candidates = candidates_by_task.get(str(task).lower() if task else "", ["target_value"])
    for c in candidates:
        if c in col_set:
            return c
    return None


def load_task_records(dataset_csv: Union[str, Path], task: Optional[str] = None) -> Dict[str, TaskRecord]:
    df = pd.read_csv(dataset_csv)
    required = {"accession", "sec_structure"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {dataset_csv}: {sorted(missing)}")

    target_col = _detect_target_column(df.columns, task)

    out: Dict[str, TaskRecord] = {}
    for _, row in df.iterrows():
        accession = str(row["accession"])
        target_value = _coerce_float_or_none(row[target_col]) if target_col else None
        out[accession] = TaskRecord(accession=accession, sec_structure=str(row["sec_structure"]), target_value=target_value)
    return out


def site_role_from_description(desc: str) -> str:
    d = (desc or "").lower()
    if "catalytic" in d or "active site" in d:
        return "catalytic"
    if "binding" in d:
        return "binding"
    return "other"


def load_functional_annotations(functional_sites_long_csv: Union[str, Path]) -> Tuple[Dict[str, Set[int]], Dict[str, Dict[int, str]]]:
    df = pd.read_csv(functional_sites_long_csv)
    required = {"accession", "pos", "description"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {functional_sites_long_csv}: {sorted(missing)}")

    sites_by_accession: Dict[str, Set[int]] = defaultdict(set)
    role_map: Dict[str, Dict[int, str]] = defaultdict(dict)
    priority = {"catalytic": 3, "binding": 2, "other": 1}

    for _, row in df.iterrows():
        accession = str(row["accession"])
        pos = int(row["pos"])
        role = site_role_from_description(str(row.get("description", "")))

        sites_by_accession[accession].add(pos)
        prev_role = role_map[accession].get(pos)
        if prev_role is None or priority[role] > priority[prev_role]:
            role_map[accession][pos] = role

    return {k: set(v) for k, v in sites_by_accession.items()}, {k: dict(v) for k, v in role_map.items()}


def split_positions_by_role(accession: str, all_positions: Set[int], role_map: Mapping[str, Mapping[int, str]]) -> Dict[str, Set[int]]:
    out = {"catalytic": set(), "binding": set(), "other": set()}
    pos2role = role_map.get(accession, {})
    for pos in all_positions:
        out[pos2role.get(pos, "other")].add(pos)
    return out


def _parse_seqres(pdb_lines: Iterable[str], chain_id: Optional[str]) -> str:
    residues: list[str] = []
    for line in pdb_lines:
        if not line.startswith("SEQRES"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        chain = parts[2]
        if chain_id is not None and chain != chain_id:
            continue
        residues.extend(parts[4:])

    return "".join(AA3_TO_1.get(res.upper(), "X") for res in residues)


def _parse_atom_fallback(pdb_lines: Iterable[str], chain_id: Optional[str]) -> str:
    seen = set()
    residues: list[str] = []

    for line in pdb_lines:
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        chain = line[21].strip() or "A"
        if chain_id is not None and chain != chain_id:
            continue
        resseq = line[22:27]
        key = (chain, resseq)
        if key in seen:
            continue
        seen.add(key)
        resname = line[17:20].strip().upper()
        residues.append(AA3_TO_1.get(resname, "X"))

    return "".join(residues)


def load_native_sequence_from_pdb(pdb_path: Union[str, Path], chain_id: Optional[str] = None) -> str:
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(str(pdb_path))

    lines = pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    seqres_seq = _parse_seqres(lines, chain_id)
    if seqres_seq:
        return normalize_sequence(seqres_seq)

    atom_seq = _parse_atom_fallback(lines, chain_id)
    if atom_seq:
        return normalize_sequence(atom_seq)

    raise ValueError(f"Could not parse sequence from PDB: {pdb_path}")

