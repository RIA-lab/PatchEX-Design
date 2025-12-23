#!/usr/bin/env python3
"""
sample_from_logits_and_pssm.py
Combine structure logits (from inverse folding) and PSI-BLAST PSSM directly (.pssm file)
to generate a small ensemble of sequences reflecting both structure and function constraints.

Usage:
  python sample_from_logits_and_pssm.py --logits path/to/logits.pt \
      --pssm BLAST/Q29495/seed.pssm --out combined_seeds.fasta --k 8
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import random
import re

AA_TYPE = ['A','R','N','D','C','Q','E','G','H','I',
           'L','K','M','F','P','S','T','W','Y','V']

# -------------------------------------------------------------
# Parse PSI-BLAST ASCII PSSM
# -------------------------------------------------------------
def parse_pssm_ascii(pssm_path: str, normalize: bool = True) -> np.ndarray:
    """
    Parse PSI-BLAST ASCII PSSM file into [L, 20] numpy array.

    Args:
        pssm_path: path to .pssm file
        normalize: convert scores to probabilities (softmax per row)

    Returns:
        arr: np.ndarray (L, 20)
    """
    with open(pssm_path) as f:
        lines = f.readlines()

    # Find header line containing A R N D ...
    header_idx = None
    for i, line in enumerate(lines):
        if re.search(r'\bA\b.*\bR\b.*\bN\b.*\bD\b', line):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Cannot find amino acid header line in {pssm_path}")

    data = []
    for line in lines[header_idx+1:]:
        if not line.strip():
            break
        toks = line.split()
        # Expect at least 22 tokens (index, residue, 20 scores, ... percentages)
        if len(toks) < 22:
            continue
        try:
            scores = list(map(float, toks[2:22]))
            data.append(scores)
        except ValueError:
            continue
        # stop if we reach the summary line
        if line.strip().startswith("Lambda") or line.strip().startswith("Psi"):
            break

    arr = np.array(data, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"No valid PSSM rows parsed from {pssm_path}")

    if normalize:
        arr = arr - arr.max(axis=1, keepdims=True)
        arr = np.exp(arr)
        arr = arr / arr.sum(axis=1, keepdims=True)
    return arr


# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------
def softmax_probs(logits: torch.Tensor, temperature: float=1.0) -> np.ndarray:
    return F.softmax(logits / temperature, dim=-1).cpu().numpy()

def entropy_per_row(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log2(p), axis=1)

def normalize01(x: np.ndarray):
    lo, hi = x.min(), x.max()
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def compute_position_weights(p_struct: np.ndarray, p_evol: np.ndarray,
                             method: str='entropy', bias: float=0.0) -> np.ndarray:
    if method == 'fixed':
        alpha = np.full((p_struct.shape[0],), 0.5 + bias)
        return np.clip(alpha, 0.0, 1.0)
    Hs = entropy_per_row(p_struct)
    He = entropy_per_row(p_evol)
    Hs_n, He_n = normalize01(Hs), normalize01(He)
    score = (He_n - Hs_n) + bias
    alpha = 1.0 / (1.0 + np.exp(-5.0 * score))
    return alpha

def combine_logspace(p_struct: np.ndarray, p_evol: np.ndarray, alpha_pos: np.ndarray) -> np.ndarray:
    eps = 1e-12
    log_s = np.log(np.clip(p_struct, eps, 1.0))
    log_e = np.log(np.clip(p_evol, eps, 1.0))
    alpha_pos = alpha_pos.reshape(-1,1)
    log_comb = alpha_pos * log_s + (1.0 - alpha_pos) * log_e
    log_comb -= log_comb.max(axis=1, keepdims=True)
    comb = np.exp(log_comb)
    comb /= comb.sum(axis=1, keepdims=True)
    return comb

def top_p_sample(probs_row: np.ndarray, top_p: float=0.9) -> int:
    idx = np.argsort(probs_row)[::-1]
    ps = probs_row[idx]
    cumsum = np.cumsum(ps)
    cutoff = np.searchsorted(cumsum, top_p, side='right')
    keep_idx = idx[:max(1, cutoff+1)]
    keep_probs = probs_row[keep_idx] / probs_row[keep_idx].sum()
    return np.random.choice(keep_idx, p=keep_probs)

def sample_sequences_combined(p_comb: np.ndarray, k: int=8, temperature: float=1.0, top_p: float=0.9, seed: int=42):
    np.random.seed(seed)
    random.seed(seed)
    L = p_comb.shape[0]
    seqs = []
    for _ in range(k):
        chars = []
        for i in range(L):
            probs = p_comb[i].copy()
            if temperature != 1.0:
                probs = np.power(probs, 1.0/temperature)
                probs = probs / probs.sum()
            aa_idx = top_p_sample(probs, top_p=top_p)
            chars.append(AA_TYPE[int(aa_idx)])
        seqs.append("".join(chars))
    return seqs

def save_fasta(seqs, out_path, prefix):
    with open(out_path, 'w') as f:
        for i,s in enumerate(seqs):
            f.write(f">{prefix}_{i+1}\n{s}\n")
    print(f"✅ Saved {len(seqs)} sequences to {out_path}")

# -------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------
def sampling(
        logits,
        pssm,
        prefix,
        k=1,
        tau_struct=0.8,
        tau_final=0.8,
        top_p=0.9,
        weight_method='entropy',
        bias=0.0,
        seed=42
):
    """
    Sample sequences from combined structural and evolutionary models.

    Args:
        logits: Path to torch logits file (required)
        pssm: Path to PSSM file (required)
        out: Output FASTA file path
        k: Number of sequences to generate
        tau_struct: Temperature for structural probabilities
        tau_final: Temperature for final sampling
        top_p: Nucleus sampling threshold
        weight_method: Weight computation method ('entropy' or 'fixed')
        bias: Bias term for weight computation
        seed: Random seed
        prefix: Prefix for sequence IDs

    Returns:
        List of generated sequences
    """
    # --- Load structure logits ---
    logits = torch.load(logits, map_location='cpu')
    logits = logits.float()

    # --- Parse PSSM directly ---
    pssm = parse_pssm_ascii(pssm, normalize=True)

    # --- Convert to probabilities ---
    p_struct = softmax_probs(logits, temperature=tau_struct)
    p_evol = pssm

    # --- Compute adaptive weights ---
    alpha_pos = compute_position_weights(p_struct, p_evol, method=weight_method, bias=bias)

    # --- Combine ---
    p_comb = combine_logspace(p_struct, p_evol, alpha_pos)

    # --- Sample ---
    seqs = []
    cons = "".join([AA_TYPE[int(np.argmax(row))] for row in p_comb])
    seqs.append(cons)

    if k - 1 > 0:
        stoch = sample_sequences_combined(p_comb, k=k - 1, temperature=tau_final, top_p=top_p, seed=seed)
        seqs.extend(stoch)
        accessions = [f'{prefix}_{i+1}' for i in range(len(seqs))]
    else:
        accessions = [f'{prefix}']

    # --- Save ---
    # save_fasta(seqs, out, prefix)

    return p_comb, accessions, seqs
