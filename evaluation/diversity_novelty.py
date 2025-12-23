import os
import subprocess
import tempfile
import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# ============================================================
# 1. Mean Pairwise Sequence Identity (Diversity)
# ============================================================

def compute_sequence_identity_mmseqs(seqs, tmp_dir="tmp_mmseqs", threads=8):
    """
    Compute mean pairwise identity (%) among generated sequences using MMseqs2.
    Returns 1 - mean_identity as diversity score (0–1 scale).
    """
    os.makedirs(tmp_dir, exist_ok=True)
    fasta_path = os.path.join(tmp_dir, "seqs.fasta")

    with open(fasta_path, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">seq{i}\n{s}\n")

    db = os.path.join(tmp_dir, "db")
    aln = os.path.join(tmp_dir, "aln")

    subprocess.run(["mmseqs", "createdb", fasta_path, db], check=True)
    subprocess.run(["mmseqs", "search", db, db, aln, os.path.join(tmp_dir, "tmp"),
                    "--min-seq-id", "0.0", "-s", "7.5", "--max-seqs", "10000", "--threads", str(threads)], check=True)

    result_path = os.path.join(tmp_dir, "pairwise.tsv")
    subprocess.run(["mmseqs", "convertalis", db, db, aln, result_path, "--format-output", "query,target,pident"], check=True)

    identities = []
    with open(result_path) as f:
        for line in f:
            q, t, pid = line.strip().split("\t")
            if q != t:
                identities.append(float(pid) / 100.0)

    mean_identity = np.mean(identities) if identities else 0.0
    diversity = 1.0 - mean_identity
    return diversity, mean_identity


# ============================================================
# 2. Novelty: % sequences <70% identity vs. training DB
# ============================================================

def compute_novelty_mmseqs(generated, training, tmp_dir="tmp_novelty", threshold=0.7, threads=8):
    """
    Fraction of generated sequences with < threshold identity to any training sequence.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    gen_fa = os.path.join(tmp_dir, "gen.fasta")
    train_fa = os.path.join(tmp_dir, "train.fasta")

    with open(gen_fa, "w") as f:
        for i, s in enumerate(generated):
            f.write(f">gen{i}\n{s}\n")
    with open(train_fa, "w") as f:
        for i, s in enumerate(training):
            f.write(f">train{i}\n{s}\n")

    gen_db, train_db, aln = [os.path.join(tmp_dir, x) for x in ["genDB", "trainDB", "aln"]]
    subprocess.run(["mmseqs", "createdb", gen_fa, gen_db], check=True)
    subprocess.run(["mmseqs", "createdb", train_fa, train_db], check=True)
    subprocess.run(["mmseqs", "search", gen_db, train_db, aln, os.path.join(tmp_dir, "tmp"),
                    "--min-seq-id", "0.0", "-s", "7.5", "--max-seqs", "10000", "--threads", str(threads)], check=True)

    result_path = os.path.join(tmp_dir, "gen_vs_train.tsv")
    subprocess.run(["mmseqs", "convertalis", gen_db, train_db, aln, result_path,
                    "--format-output", "query,target,pident"], check=True)

    best_hits = {}
    with open(result_path) as f:
        for line in f:
            q, t, pid = line.strip().split("\t")
            pid = float(pid) / 100.0
            best_hits[q] = max(best_hits.get(q, 0.0), pid)

    below_thresh = [q for q, pid in best_hits.items() if pid < threshold]
    novelty = len(below_thresh) / len(generated)
    return novelty


# ============================================================
# 3. Shannon Entropy / Perplexity
# ============================================================

def compute_entropy_perplexity(seqs):
    """
    Compute average Shannon entropy and perplexity over positions.
    """
    if not seqs:
        return 0, 0
    L = max(len(s) for s in seqs)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}
    counts = np.zeros((L, 20))

    for s in seqs:
        for i, aa in enumerate(s):
            if aa in aa_to_idx:
                counts[i, aa_to_idx[aa]] += 1

    probs = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    entropy = -np.nansum(probs * np.log2(np.clip(probs, 1e-9, 1)), axis=1)
    avg_entropy = np.mean(entropy)
    perplexity = np.mean(2 ** entropy)
    return float(avg_entropy), float(perplexity)


# ============================================================
# 4. Embedding Diversity (ESM-2)
# ============================================================

def compute_embedding_diversity_esm(seqs, model_name="facebook/esm2_t33_650M_UR50D", device=None):
    """
    Compute average pairwise cosine distance between ESM-2 embeddings.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    for seq in seqs:
        inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            outputs = model(**{k: v.to(device) for k, v in inputs.items()})
        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(emb)

    embs = np.vstack(embeddings)
    sim = cosine_similarity(embs)
    dist = 1 - sim
    tril_idx = np.tril_indices_from(dist, k=-1)
    mean_dist = np.mean(dist[tril_idx])
    return float(mean_dist)
