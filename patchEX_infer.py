import yaml
from utils import load_weight
from models import load_model
import torch
from dataclasses import dataclass
from typing import Any
import numpy as np


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
    def __init__(self, checkpoint_dir, checkpoint):
        model_config_path = f'{checkpoint_dir}/model_config.yaml'
        # model_config_path = 'model_configs/PatchEX.yaml'
        weight_path = f'{checkpoint_dir}/checkpoint-{checkpoint}/model.safetensors'

        with open(model_config_path, 'r') as file:
            model_config = yaml.safe_load(file)

        Model, Collator = load_model(model_config['name'])
        self.collate_fn = Collator(model_config['pretrain_model'])
        self.model = Model(model_config)
        load_weight(self.model, weight_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.inference = True
        self.model.eval()

    def inference(self, data):
        batch = self.collate_fn(data)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
        for idx, item in enumerate(data):
            item.label = outputs.pred[idx].cpu().item()
            item.weights = outputs.patch_weights[idx].cpu()
        return data


if __name__ == '__main__':
    task = 'ph'
    output_dir = f'output/{task}/PatchEX'
    checkpoint_point = '34419' if task == 'opt' else '5280'
    model = InferenceModel(output_dir, checkpoint_point)
    data = [SeqItem(accession='Q29495_1', sequence='MIKIILLALMALLLPTLAQAPSTAVYPIKKQNNCQDRTNVTLRKLRSKEFGNPCQFENQPLLIVNISSNCGFTPQFAGLEAVYNKYKDQGLVVLGFPSDDFFQEENNEQDTAKVCFVNYGVTFTMFATSAVRGSDANPIFKHLNSQTSSPNWNFYKYLVSADRKTITRRPGSYEVCEFESEKGTSAKPATLAIN'),
            SeqItem(accession='A6YGF1', sequence='MPQSKSRKIAILGYRSVGKSSLTIQFVEGQFVDSYDPTIENTFTKLITVNGQEYHLQLVDTAGQDEYSIFPQTYSIDINGYILVYSVTSIKSFEVIKVIHGKLLDMVGKVQIPIMLVGNKKDLHMERVISYEEGKALAESWNAAFLESSAKENQTAVDVFRRIILEAEKMDGAASQGKSSCSVM')]
    data = model.inference(data)
    for item in data:
        item.map_weights()
        print(f'Accession: {item.accession}, Predicted Value: {item.label}')
        print('Patch Weights:')
        print(len(item.weights), len(item.sequence))
        print(item.weights[::25])