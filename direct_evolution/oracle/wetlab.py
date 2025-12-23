from dataclasses import dataclass
import numpy as np
from typing import Any
import math


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


class WetLab_Landscape:
    def __init__(self, oracle_model, wt_accession, selected_residue_idx, wt_seq, target_value, sigma=1.0):
        self.oracle_model = oracle_model
        self.wt_accession = wt_accession
        self.selected_residue_idx = selected_residue_idx
        self.wt_seq = wt_seq
        self.target_value = target_value
        self.sigma = sigma
        seq_data = SeqItem(accession=wt_accession, sequence=wt_seq)
        seq_data = oracle_model.inference([seq_data])
        self.wt_fitness = self.fitness(seq_data[0].label)

    
    def is_valid_seq(self, sequence):
        return True

    def get_wt_fitness(self):
        return self.wt_fitness

    def map_mutated_residues(self, mutated_residues):
        mapped_seqs = []
        for seq in mutated_residues:
            full_seq = list(self.wt_seq)
            for i, idx in enumerate(self.selected_residue_idx):
                full_seq[idx] = seq[i]
            mapped_seqs.append(''.join(full_seq))
        return mapped_seqs

    def fitness(self, pred_value):
        delta = abs(pred_value - self.target_value)
        fitness = math.exp(-(delta ** 2) / (2 * self.sigma ** 2))
        return round(fitness, 2)

    def get_fitness(self, mutated_residues):
        mutated_seqs = self.map_mutated_residues(mutated_residues)
        mutated_seq_data = [SeqItem(accession=self.wt_accession, sequence=seq) for seq in mutated_seqs]
        mutated_seq_data = self.oracle_model.inference(mutated_seq_data)
        mutated_seq_data_labels = [item.label for item in mutated_seq_data]
        fitness_scores = [self.fitness(_) for _ in mutated_seq_data_labels]
        return fitness_scores
