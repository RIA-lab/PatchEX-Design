import torch
from safetensors import safe_open
import numpy as np
import json
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize
from scipy.stats import rankdata
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import yaml
import requests
from time import sleep
from tqdm import tqdm


def load_config(config_path):
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    return configs

def save_config(configs, config_path):
    with open(config_path, 'w') as file:
        yaml.dump(configs, file, default_flow_style=False, sort_keys=False)



def read_fasta(fasta, return_as_dict=False):
    headers, sequences = [], []
    with open(fasta, 'r') as fast:

        for line in fast:
            if line.startswith('>'):
                head = line.replace('>', '').strip()
                headers.append(head)
                sequences.append('')
            else:
                seq = line.strip()
                if len(seq) > 0:
                    sequences[-1] += seq

    if return_as_dict:
        return dict(zip(headers, sequences))
    else:
        return (headers, sequences)


def write_fasta(headers, seqdata, path):
    with open(path, 'w') as pp:
        for i in range(len(headers)):
            pp.write('>' + headers[i] + '\n' + seqdata[i] + '\n')
    return


def map_mutated_residues(selected_residue_idx, mutated_residues, wt_seq):
    mapped_seqs = []
    for seq in mutated_residues:
        full_seq = list(wt_seq)
        for i, idx in enumerate(selected_residue_idx):
            full_seq[idx] = seq[i]
        mapped_seqs.append(''.join(full_seq))
    return mapped_seqs

def get_raw_accession(s):
    return s.split('_')[0]

def read_json(path):
    f = open(path, 'r')
    readdict = json.load(f)
    f.close()
    return readdict

def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def write_json(writedict, path, indent=4, sort_keys=False):
    f = open(path, 'w')
    _ = f.write(json.dumps(writedict, indent=indent, sort_keys=sort_keys, default=convert_to_serializable))
    f.close()

# freeze the model parameters
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


#count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_weight(model, checkpoint_path, strict=False):
    state_dict = {}
    with safe_open(checkpoint_path, 'pt') as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    model.load_state_dict(state_dict, strict=strict)


def load_weight_part(model, checkpoint_path, part):
    state_dict = {}
    with safe_open(checkpoint_path, 'pt') as f:
        for key in f.keys():
            if part == key.split('.')[0]:
                state_dict[key] = f.get_tensor(key)
    print(state_dict.keys())
    model.load_state_dict(state_dict, strict=False)



def metrics_reg(eval_pred):
    predictions, targets = eval_pred
    # R? (Coefficient of Determination)
    r2 = r2_score(targets, predictions)
    # Pearson Correlation Coefficient
    pearson_corr = np.corrcoef(predictions, targets)[0, 1]
    # Spearman Correlation Coefficient
    spearman_corr = np.corrcoef(rankdata(predictions), rankdata(targets))[0, 1]
    # MAE
    mae = np.mean(np.abs(predictions - targets))
    #RMSE
    rmse = np.sqrt(mean_squared_error(targets, predictions))

    return {
        'R2': float(r2),
        'Pearson Correlation': float(pearson_corr),
        'Spearman Correlation': float(spearman_corr),
        'MAE': float(mae),
        'RMSE': float(rmse)
    }


def metrics_range(eval_pred):
    """
    Compute the mean overlap ratio between the predicted and true temperature ranges using NumPy.

    Args:
        eval_pred: Tuple (pred, label)
            - pred: ndarray of shape (batch_size, 2), where pred[:, 0] is T_low and pred[:, 1] is T_high
            - label: ndarray of shape (batch_size, 2), where label[:, 0] is T_low and label[:, 1] is T_high

    Returns:
        dict: {'mean_overlap_ratio': mean_overlap_ratio}
    """
    pred, label = eval_pred
    pred_low, pred_high = pred[:, 0], pred[:, 1]
    label_low, label_high = label[:, 0], label[:, 1]

    mae_low = np.mean(np.abs(pred_low - label_low))
    mae_high = np.mean(np.abs(pred_high - label_high))

    # Calculate intersection range
    inter_low = np.maximum(pred_low, label_low)
    inter_high = np.minimum(pred_high, label_high)

    # Calculate overlap length (if inter_high > inter_low, otherwise zero)
    intersection = np.maximum(inter_high - inter_low, 0)

    # Calculate ground truth range length
    label_range = label_high - label_low

    # Avoid division by zero (if label_range is zero, set overlap to zero)
    overlap_ratio = np.where(label_range > 0, intersection / label_range, 0)

    # Compute mean overlap ratio
    mean_overlap_ratio = np.mean(overlap_ratio)
    return {'mean_overlap_ratio': mean_overlap_ratio, 'mae_low': mae_low, 'mae_high': mae_high}

def overlap_ratio(pred, label, save_dir):
    pred_low, pred_high = pred[:, 0], pred[:, 1]
    label_low, label_high = label[:, 0], label[:, 1]
    # Calculate intersection range
    inter_low = np.maximum(pred_low, label_low)
    inter_high = np.minimum(pred_high, label_high)

    # Calculate overlap length (if inter_high > inter_low, otherwise zero)
    intersection = np.maximum(inter_high - inter_low, 0)

    # Calculate ground truth range length
    label_range = label_high - label_low

    # Avoid division by zero (if label_range is zero, set overlap to zero)
    overlap_ratio = np.where(label_range > 0, intersection / label_range, 0)

    # Save the overlap ratio to a file
    np.save(f'{save_dir}/overlap_ratio.npy', overlap_ratio)
    return overlap_ratio

def load_metrics(name):
    if name == 'opt':
        return metrics_reg
    elif name == 'stability':
        return metrics_reg
    elif name == 'range':
        return metrics_range
    else:
        raise ValueError('Invalid dataset name')


def scatter_plot_with_density(x, y, xlabel, ylabel, save_dir, title='Scatter_Plot'):
    # Calculate point density
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)

    # Sort the points by density, so higher density points are plotted on top
    idx = density.argsort()
    x, y, density = x[idx], y[idx], density[idx]

    # Plot the scatter plot with density colormap
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=density, s=20, cmap='viridis')
    plt.colorbar(label='Density')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([np.min(x), np.max(x)])
    plt.ylim([np.min(x), np.max(x)])
    plt.title(title)
    if 'PatchET' in title:
        title = 'MAE_low' if 'low' in title else 'MAE_high'
    plt.savefig(f'{save_dir}/{title}.pdf', bbox_inches='tight')


def interval_evaluation_opt(labels, predictions):
    index_25 = np.where(labels < 25)[0]
    index25_50 = np.where((labels >= 25) & (labels < 50))[0]
    index50_80 = np.where((labels >= 50) & (labels < 80))[0]
    index80_ = np.where(labels>=80)[0]
    index_dict = {'<25': index_25, '25-50': index25_50, '50-80': index50_80, '>80': index80_}
    metrics_interval = {}
    for k, v in index_dict.items():
        predictions_interval = predictions[v]
        labels_interval = labels[v]
        metrics = metrics_reg((predictions_interval, labels_interval))
        metrics_interval[k] = metrics
    return metrics_interval


def interval_evaluation_stability(labels, predictions):
    index_45 = np.where(labels < 45)[0]
    index45_70 = np.where((labels >= 45) & (labels < 70))[0]
    index70_ = np.where(labels >= 70)[0]

    index_dict = {'<45': index_45, '45-70': index45_70, '>70': index70_}
    metrics_interval = {}
    for k, v in index_dict.items():
        predictions_interval = predictions[v]
        labels_interval = labels[v]
        metrics = metrics_reg((predictions_interval, labels_interval))
        metrics_interval[k] = metrics
    return metrics_interval


def plot_interval_evaluation(metrics_interval, save_dir):
    x = metrics_interval.keys()
    y = [metrics_interval[k]['MAE'] for k in x]
    plt.figure()
    plt.bar(x, y)

    # Display the value of the bar
    for i, v in enumerate(y):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')

    plt.xlabel('Temperature Interval')
    plt.ylabel('MAE')
    plt.title('MAE for different temperature intervals')
    plt.savefig(f'{save_dir}/MAE_for_different_temperature_intervals.png')


def download_swissprot_by_ec(ec_number: str,
                             output_fasta_path: str,
                             batch_size: int = 500,
                             max_retries: int = 3,
                             pause_between: float = 1.0):
    """
    Download all Swiss-Prot (reviewed) sequences for a given EC number from UniProt,
    and save them as a FASTA file with a progress bar.

    Args:
        ec_number: e.g. "1.1.1.1" — the EC number to query.
        output_fasta_path: local file path to write the FASTA sequences.
        batch_size: how many entries to request per page.
        max_retries: number of times to retry failed requests.
        pause_between: seconds to wait between successive API calls.
    """

    # Base URLs
    search_url = "https://rest.uniprot.org/uniprotkb/search"

    # Step 1: find total number of hits (JSON format)
    count_params = {
        "query": f"ec:{ec_number} AND reviewed:true",
        "format": "json",
        "size": 0  # don't fetch entries, just metadata
    }
    response = requests.get(search_url, params=count_params, timeout=30)
    response.raise_for_status()
    total_hits = response.json()["total"]
    print(f"Found {total_hits} Swiss-Prot entries for EC {ec_number}.")

    if total_hits == 0:
        print("No sequences found.")
        return

    # Step 2: fetch in FASTA batches
    params = {
        "query": f"ec:{ec_number} AND reviewed:true",
        "format": "fasta",
        "size": batch_size,
    }

    fasta_entries = []
    next_url = None

    with tqdm(total=total_hits, desc="Downloading sequences", unit="seq") as pbar:
        # First request
        for attempt in range(max_retries):
            try:
                response = requests.get(search_url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.text
                    fasta_entries.append(data)
                    # Count sequences in this batch (lines starting with ">")
                    seq_count = data.count(">")
                    pbar.update(seq_count)
                    break
                else:
                    print(f"Attempt {attempt + 1}: HTTP {response.status_code}; retrying …")
            except Exception as e:
                print(f"Attempt {attempt + 1}: error {e}; retrying …")
            sleep(pause_between)
        else:
            raise RuntimeError("Failed to fetch initial data from UniProt for EC " + ec_number)

        # Pagination loop
        while True:
            link_header = response.headers.get("Link")
            next_url = None
            if link_header:
                for part in link_header.split(","):
                    section = part.strip().split(";")
                    if len(section) < 2:
                        continue
                    url_part = section[0].strip()
                    rel_part = section[1].strip()
                    if rel_part == 'rel="next"':
                        if url_part.startswith("<") and url_part.endswith(">"):
                            next_url = url_part[1:-1]
                        else:
                            next_url = url_part
                        break

            if not next_url:
                break

            # Fetch next page
            for attempt in range(max_retries):
                try:
                    response = requests.get(next_url, timeout=30)
                    if response.status_code == 200:
                        data = response.text
                        fasta_entries.append(data)
                        seq_count = data.count(">")
                        pbar.update(seq_count)
                        break
                    else:
                        print(f"Next page attempt {attempt + 1}: HTTP {response.status_code}; retrying …")
                except Exception as e:
                    print(f"Next page attempt {attempt + 1}: error {e}; retrying …")
                sleep(pause_between)
            else:
                print("Warning: failed to fetch next page, stopping early.")
                break

    # Save to file
    with open(output_fasta_path, "w") as f:
        for entry in fasta_entries:
            f.write(entry)

    print(f"✅ Download complete. Sequences saved to {output_fasta_path}")


