import os
from sample_from_logits_and_pssm import sampling
import subprocess
from patchEX_infer import InferenceModel
from dataclasses import dataclass
import numpy as np
from typing import Any


@dataclass
class SeqItem:
    accession: str
    sequence: str
    label: float = 0
    weights: Any = None  # ✅ Fixed: proper type annotation
    idx: list = None
    score: np.ndarray = None

    def map_weights(self):
        length = len(self.sequence)
        last_patch_idx = length // 25
        last_patch_residue = length % 25
        weights = []
        for i in range(last_patch_idx):
            weights.extend([self.weights[i].item()] * 25)
        if last_patch_residue > 0:
            weights.extend([self.weights[last_patch_idx].item()] * last_patch_residue)
        self.weights = np.asarray(weights)


def extract_pred_to_fasta(pred_file_path):
    """
    Extract the predicted sequence from a pred.txt file and save it as a FASTA file.

    Args:
        pred_file_path (str): Path to the pred.txt file (e.g., 'MapDiffInferResults/P9WMP9.pdb/pred.txt')

    Returns:
        str: Path to the created FASTA file, or None if failed
    """
    try:
        # Extract protein ID from the directory name
        dir_name = os.path.basename(os.path.dirname(pred_file_path))
        protein_id = dir_name

        # Read the pred.txt file
        with open(pred_file_path, 'r') as f:
            lines = f.readlines()

        # Find the predicted sequence
        pred_sequence = None
        in_pred_section = False

        for line in lines:
            line = line.strip()
            if line == '>pred':
                in_pred_section = True
                continue
            elif line.startswith('>') and in_pred_section:
                # End of pred section
                break
            elif in_pred_section and line:
                # This is the predicted sequence
                pred_sequence = line
                break

        if pred_sequence is None:
            print(f"Error: Could not find predicted sequence in {pred_file_path}")
            return None

        # Create output FASTA file
        output_dir = os.path.dirname(pred_file_path)
        output_file = os.path.join(output_dir, f"{protein_id}.fasta")

        with open(output_file, 'w') as f:
            f.write(f">{protein_id}\n")
            f.write(f"{pred_sequence}\n")

        print(f"Successfully created {output_file}")
        return output_file

    except FileNotFoundError:
        print(f"Error: File {pred_file_path} not found")
        return None
    except IOError as e:
        print(f"Error reading/writing file: {e}")
        return None

def merge_fasta_files(input_files, output_file):
    """
    Merge multiple FASTA files into a single FASTA file.

    Args:
        input_files (list): List of paths to input FASTA files
        output_file (str): Path to the output merged FASTA file

    Returns:
        str: output_file
    """
    sequence_count = 0

    try:
        with open(output_file, 'w') as outfile:
            for file_path in input_files:
                try:
                    with open(file_path, 'r') as infile:
                        content = infile.read()
                        # Add newline between files if the previous file doesn't end with one
                        if content and not content.endswith('\n'):
                            content += '\n'
                        outfile.write(content)

                        # Count sequences in this file (lines starting with '>')
                        sequence_count += content.count('>')

                except FileNotFoundError:
                    print(f"Warning: File {file_path} not found, skipping...")
                except IOError as e:
                    print(f"Warning: Error reading {file_path}: {e}, skipping...")

    except IOError as e:
        print(f"Error: Cannot write to output file {output_file}: {e}")
        return 0

    print(f"Successfully merged {len(input_files)} files into {output_file}")
    print(f"Total sequences: {sequence_count}")
    return output_file

def run_psiblast(pred_fasta, ec_fasta, threads=8, show_realtime=True, output_dir="BLAST"):
    """
    Run PSI-BLAST against EC-specific pool.

    Args:
        pred_fasta (str): path to predicted fasta from inverse folding model
        ec_fasta (str): path to EC fasta (e.g. msa_cache/2_3_1_87/2_3_1_87.fasta)
        threads (int): number of threads for PSI-BLAST
        show_realtime (bool): whether to stream log output
        output_dir (str): base directory for BLAST output (default: "BLAST")

    Returns:
        str: path to generated PSSM file, or None if failed
    """
    env = os.environ.copy()
    env["THREADS"] = str(threads)
    # Pass output_dir as third argument to blast.sh
    cmd = ["bash", "blast.sh", pred_fasta, ec_fasta, output_dir]

    try:
        if show_realtime:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, env=env)
            for line in process.stdout:
                print(line, end="")
            process.wait()
            if process.returncode == 0:
                acc_name = os.path.splitext(os.path.basename(pred_fasta))[0]
                return f"{output_dir}/{acc_name}/seed.pssm"
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            if result.returncode == 0:
                acc_name = os.path.splitext(os.path.basename(pred_fasta))[0]
                return f"{output_dir}/{acc_name}/seed.pssm"
    except Exception as e:
        print(f"[ERROR] PSI-BLAST failed: {e}")
    return None

def select_flexible_residues(p_comb: np.ndarray, weights: np.ndarray,
                             k: int = 8, beta: float = 0.5,
                             exclude_idx: list = None):
    """
    Select flexible amino acid positions for design.

    Args:
        p_comb (np.ndarray): combined probability matrix [L, 20]
        weights (np.ndarray): contribution scores [L]
        k (int): number of residues to select
        beta (float): weight for entropy vs contribution (0.5 = equal)
        exclude_idx (list): optional indices to exclude (e.g. catalytic sites)

    Returns:
        selected_idx (list): indices (0-based) of selected residues
        score_table (np.ndarray): [L, 3] array of H_norm, ph_norm, final score
    """
    L = p_comb.shape[0]
    eps = 1e-12

    # 1. Entropy per position
    p = np.clip(p_comb, eps, 1.0)
    H = -np.sum(p * np.log2(p), axis=1)  # [L]

    # 2. Normalize entropy and pH weights
    H_norm = (H - H.min()) / (H.max() - H.min() + 1e-8)
    C = np.array(weights)
    C_norm = (C - C.min()) / (C.max() - C.min() + 1e-8)

    # 3. Combine
    score = (H_norm ** beta) * (C_norm ** (1 - beta))

    # 4. Exclude protected sites (e.g. catalytic)
    if exclude_idx is not None and len(exclude_idx) > 0:
        score[exclude_idx] = -np.inf  # do not select these

    # 5. Rank and select
    selected_idx = np.argsort(score)[::-1][:k]

    score_table = np.vstack([H_norm, C_norm, score]).T
    return selected_idx.tolist(), score_table


class InitialSampler:
    def __init__(self, oracle_model: InferenceModel):
        self.oracle_model = oracle_model

    def sample_seq(self, accession, ec):
        pred_fasta = extract_pred_to_fasta(f'MapDiffInferResults/{accession}/pred.txt')
        ec_cache = f'msa_cache/{ec.replace(".", "_")}'
        if not os.path.exists(f'{ec_cache}/{ec.replace(".", "_")}.fasta'):
            ec_fasta_files = [os.path.join(ec_cache, f) for f in os.listdir(ec_cache) if f.endswith(('.fasta', '.fa'))]
            merged_fasta = f'{ec_cache}/{ec.replace(".", "_")}.fasta'
            ec_pool_fasta = merge_fasta_files(ec_fasta_files, merged_fasta)
        else:
            ec_pool_fasta = f'{ec_cache}/{ec.replace(".", "_")}.fasta'

        pssm_path = run_psiblast(pred_fasta, ec_pool_fasta, threads=8, show_realtime=True)
        if not pssm_path or not os.path.exists(pssm_path) or os.path.getsize(pssm_path) == 0:
            print(f"[WARN] No valid PSSM found for {accession}, skipping.")
            return None, None, None

        logits = f'MapDiffInferResults/{accession}/logits.pt'
        p_comb, accessions, sampled_seqs = sampling(logits, pssm_path, accession)
        return p_comb, accessions, sampled_seqs

    def sample_idx(self, p_comb: np.ndarray, accessions: list, sampled_seqs: list, k: int =4, beta: float =0.5, exclude_idx: list =None):
        seq_data = [SeqItem(accession=acc, sequence=seq) for acc, seq in zip(accessions, sampled_seqs)]
        seq_data = self.oracle_model.inference(seq_data)
        for item in seq_data:
            item.map_weights()

        for item in seq_data:
            item.idx, item.score = select_flexible_residues(p_comb, item.weights, k=k, beta=beta, exclude_idx=exclude_idx)
            print(f'Selected positions for {item.accession}: {item.idx}')
        return seq_data



if __name__ == "__main__":
    accession = 'Q29495'
    ec = '2.3.1.87'
    task = 'opt'
    output_dir = f'output/{task}/PatchET'
    checkpoint_point = '34419' if task == 'opt' else '5280'

    pred_fasta = extract_pred_to_fasta(f'MapDiffInferResults/{accession}/pred.txt')
    ec_cache = f'msa_cache/{ec.replace(".", "_")}'
    if not os.path.exists(f'{ec_cache}/{ec.replace(".", "_")}.fasta'):
        ec_fasta_files = [os.path.join(ec_cache, f) for f in os.listdir(ec_cache) if f.endswith(('.fasta', '.fa'))]
        merged_fasta = f'{ec_cache}/{ec.replace(".", "_")}.fasta'
        ec_pool_fasta = merge_fasta_files(ec_fasta_files, merged_fasta)
    else:
        ec_pool_fasta = f'{ec_cache}/{ec.replace(".", "_")}.fasta'

    pssm_path = run_psiblast(pred_fasta, ec_pool_fasta, threads=8, show_realtime=True)
    logits = f'MapDiffInferResults/{accession}/logits.pt'
    p_comb, sampled_seqs = sampling(logits, pssm_path, f'RESULT/{accession}/sampled_seq.fasta', accession)

    model = InferenceModel(output_dir, checkpoint_point)
    seq_data = [SeqItem(accession=f'{accession}_{i+1}', sequence=seq) for i, seq in enumerate(sampled_seqs)]
    seq_data = model.inference(seq_data)
    for item in seq_data:
        item.map_weights()

    for item in seq_data:
        item.idx, item.score = select_flexible_residues(p_comb, item.weights, k=8, beta=0.5, exclude_idx=None)
        print(f'Selected positions for {item.accession}: {item.idx}')