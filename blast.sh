#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash blast.sh pred_seq.fasta msa_cache/2_3_1_87/2_3_1_87.fasta [output_dir]

QUERY_FASTA=$1      # e.g., path to predicted fasta from IPF model
EC_FASTA=$2         # e.g., path to EC sequence pool fasta
OUTPUT_BASE=${3:-BLAST}  # Optional third parameter, defaults to "BLAST"
THREADS=${THREADS:-8}

# Derive output dir
ACC_NAME=$(basename "$QUERY_FASTA" .fasta)
OUTDIR="${OUTPUT_BASE}/${ACC_NAME}"
mkdir -p "$OUTDIR"

echo "[INFO] Query fasta: $QUERY_FASTA"
echo "[INFO] EC pool fasta: $EC_FASTA"
echo "[INFO] Output dir: $OUTDIR"

# ============================================================
# Step A: Prepare BLAST database
# ============================================================
if [ ! -f "${EC_FASTA}.pin" ]; then
    echo "[Step A] Creating BLAST database..."
    makeblastdb -in "$EC_FASTA" -dbtype prot -parse_seqids -out "${EC_FASTA}"
else
    echo "[Step A] Existing BLAST database detected, skipping."
fi

# ============================================================
# Step B: Run PSI-BLAST
# ============================================================
echo "[Step B] Running PSI-BLAST..."
psiblast -query "$QUERY_FASTA" \
         -db "$EC_FASTA" \
         -num_iterations 3 \
         -evalue 1e-3 \
         -out_ascii_pssm "$OUTDIR/seed.pssm" \
         -out "$OUTDIR/psiblast_report.txt" \
         -num_threads $THREADS \
         -save_pssm_after_last_round

# ============================================================
# Step C: Check output
# ============================================================
if [ -s "$OUTDIR/seed.pssm" ]; then
    echo "[DONE] PSI-BLAST completed successfully."
    echo "       PSSM saved to $OUTDIR/seed.pssm"
else
    echo "[ERROR] PSI-BLAST failed or produced empty PSSM."
    exit 1
fi
