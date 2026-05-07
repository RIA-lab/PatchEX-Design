import subprocess
import re
import os


def get_raw_accession(s):
    return s.split('_')[0]


def sequence_recovery(native_seq: str, pred_seq: str, ignore_gaps: bool = True) -> float:
    """Compute sequence recovery (%) between native and predicted sequences."""
    native_seq = native_seq.strip().upper()
    pred_seq = pred_seq.strip().upper()
    L = min(len(native_seq), len(pred_seq))
    native_seq = native_seq[:L]
    pred_seq = pred_seq[:L]

    match = valid = 0
    for a, b in zip(native_seq, pred_seq):
        if ignore_gaps and ('-' in (a, b)):
            continue
        valid += 1
        if a == b:
            match += 1
    return (match / valid) if valid > 0 else 0.0


def compute_tmscore(pdb1: str, pdb2: str):
    """Run TM-align and parse TM-score and RMSD."""
    # Check if files exist
    if not os.path.exists(pdb1):
        print(f"Error: {pdb1} does not exist")
        return None, None
    if not os.path.exists(pdb2):
        print(f"Error: {pdb2} does not exist")
        return None, None

    try:
        cmd = ["TMalign", pdb1, pdb2]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        out = result.stdout

        # Updated regex patterns to match actual TMalign output
        # TM-score format: "TM-score= 0.99235"
        tm_match = re.search(r"TM-score=\s*([\d\.]+)", out)

        # RMSD format: "Aligned length= 256, RMSD=   0.52"
        rmsd_match = re.search(r"RMSD=\s*([\d\.]+)", out)

        tm_score = float(tm_match.group(1)) if tm_match else None
        rmsd = float(rmsd_match.group(1)) if rmsd_match else None

        if tm_score is None or rmsd is None:
            print(f"Warning: Could not parse TM-score or RMSD from output")
            print(f"TM-score match: {tm_match}, RMSD match: {rmsd_match}")

        return tm_score, rmsd

    except subprocess.CalledProcessError as e:
        print(f"TM-align command failed with return code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None, None
    except FileNotFoundError:
        print(f"Error: TMalign command not found. Is it installed and in PATH?")
        return None, None
    except subprocess.TimeoutExpired:
        print(f"Error: TMalign timed out after 60 seconds")
        return None, None
    except Exception as e:
        print(f"Unexpected error running TMalign: {type(e).__name__}: {e}")
        return None, None



