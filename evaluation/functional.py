#!/usr/bin/env python3
"""
Functional evaluation for designed enzymes:
  1. Functional embedding similarity using ESM-2.
  2. MMseqs2 hit rate against the EC-specific pool.

Each designed sequence should have a `seq_info.json` file in RESULT/**/.
"""

import os
import json
import glob
import torch
import numpy as np
import pandas as pd
import subprocess
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity


# ===========================================
# Functional Embedding Similarity
# ===========================================

def get_esm2_embedding(sequence: str, model, tokenizer, device="cuda"):
    """Compute mean-pooled ESM-2 embedding for a sequence."""
    model = model.eval()
    with torch.no_grad():
        tokens = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        outputs = model(**tokens)
        reps = outputs.last_hidden_state.squeeze(0)  # [L, D]
        mask = tokens["attention_mask"].squeeze(0)
        mean_emb = (reps * mask.unsqueeze(-1)).sum(0) / mask.sum()
    return mean_emb.cpu().numpy()


def compute_functional_embedding_similarity(seq_info_paths, ec_pool_dir="msa_cache", device="cuda"):
    """
    Compute functional embedding similarity between designed seqs and EC pool.

    For each generated sequence:
      - Get ESM-2 embedding
      - Compare to mean embedding of EC pool (from the same EC number)
    """
    print("[INFO] Loading ESM-2 model (facebook/esm2_t33_650M_UR50D)...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)

    results = []
    for json_path in tqdm(seq_info_paths, desc="Functional Embedding Similarity"):
        with open(json_path) as f:
            info = json.load(f)
        seq = info["seq"]
        accession = info["accession"]

        # Try to parse EC number from folder structure (e.g., RESULT/ph/2.7.4.7/...)
        parts = json_path.split("/")
        ec_str = None
        for p in parts:
            if p.count(".") == 3:  # crude EC pattern
                ec_str = p
                break

        if ec_str is None:
            print(f"[WARN] EC not found for {accession}, skipping embedding comparison.")
            continue

        ec_fasta = f"msa_cache/{ec_str.replace('.', '_')}/{ec_str.replace('.', '_')}.fasta"
        if not os.path.exists(ec_fasta):
            print(f"[WARN] EC pool missing for {ec_str}, skipping.")
            continue

        # Load 3–5 representative EC sequences to form mean embedding
        ec_seqs = []
        with open(ec_fasta) as f_ec:
            lines = f_ec.read().splitlines()
            for i in range(0, len(lines), 2):
                ec_seqs.append(lines[i+1].strip())
                if len(ec_seqs) >= 5:
                    break

        gen_emb = get_esm2_embedding(seq, model, tokenizer, device)
        pool_embs = [get_esm2_embedding(s, model, tokenizer, device) for s in ec_seqs]
        ec_mean = np.mean(pool_embs, axis=0)
        cos_sim = cosine_similarity([gen_emb], [ec_mean])[0][0]

        out_dir = os.path.dirname(json_path)
        out_path = os.path.join(out_dir, "functional_similarity.json")
        with open(out_path, "w") as fw:
            json.dump({"functional_embedding_similarity": float(cos_sim)}, fw, indent=4)

        results.append({
            "accession": accession,
            "ec": ec_str,
            "functional_embedding_similarity": cos_sim,
            "out_dir": out_dir
        })

    df = pd.DataFrame(results)
    df.to_csv("functional_embedding_similarity.csv", index=False)
    print("\n✅ Functional embedding similarity complete.")
    return df


