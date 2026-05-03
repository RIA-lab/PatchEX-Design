import yaml
from utils import load_weight
from models import load_model
import torch
from dataclasses import dataclass
from typing import Any
import numpy as np
import os


def resolve_runtime_device(device_preference=None):
    if isinstance(device_preference, torch.device):
        return device_preference

    if device_preference is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalized = str(device_preference).strip().lower()
    if normalized.startswith('cuda') and not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(device_preference)


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


class InferenceModel:
    def __init__(self, checkpoint_dir, checkpoint=None, device=None):
        model_config_path, weight_path = self._resolve_checkpoint_paths(checkpoint_dir, checkpoint)

        with open(model_config_path, 'r') as file:
            model_config = yaml.safe_load(file)

        Model, Collator = load_model(model_config['name'])
        self.collate_fn = Collator(model_config['pretrain_model'])
        self.model = Model(model_config)
        load_weight(self.model, weight_path)
        self.device = resolve_runtime_device(device)
        self.model.to(self.device)
        self.model.inference = True
        self.model.eval()

    @staticmethod
    def _resolve_checkpoint_paths(checkpoint_dir, checkpoint=None):
        model_config_path = os.path.join(checkpoint_dir, 'model_config.yaml')
        flat_weight_path = os.path.join(checkpoint_dir, 'model.safetensors')

        if os.path.exists(model_config_path) and os.path.exists(flat_weight_path):
            return model_config_path, flat_weight_path

        if checkpoint is not None:
            legacy_weight_path = os.path.join(checkpoint_dir, f'checkpoint-{checkpoint}', 'model.safetensors')
            if os.path.exists(model_config_path) and os.path.exists(legacy_weight_path):
                return model_config_path, legacy_weight_path

        searched_paths = [flat_weight_path]
        if checkpoint is not None:
            searched_paths.append(os.path.join(checkpoint_dir, f'checkpoint-{checkpoint}', 'model.safetensors'))

        raise FileNotFoundError(
            'Could not locate PatchEX weights. Expected one of: '
            + ', '.join(searched_paths)
        )

    def inference(self, data):
        batch = self.collate_fn(data)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
        for idx, item in enumerate(data):
            item.label = outputs.pred[idx].cpu().item()
            item.weights = outputs.patch_weights[idx].cpu()
        return data

