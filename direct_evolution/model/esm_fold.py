import torch
import esm
import os
import numpy as np
from Bio.PDB import PDBParser, Superimposer
from Bio.SVDSuperimposer import SVDSuperimposer
from tqdm import tqdm

class ESMFold:
    def __init__(self, task, wt, mut_pos, device, chunk_size=None) -> None:
        self.model = esm.pretrained.esmfold_v1().eval().to(device)
        if chunk_size is not None:
            self.model.set_chunk_size(chunk_size)
        self.wt = wt
        self.mut_pos = mut_pos
        self.wt_seg = ''.join([wt[pos] for pos in mut_pos])
        self.parser = PDBParser()
        self.task = task
    
    def get_full_seq(self, seq):
        result = list(self.wt)
        for i, pos in enumerate(self.mut_pos):
            result[pos] = seq[i]
        return ''.join(result)

    def generate_pdb(self, sequences):
        if not os.path.exists(f'pdb_cache/{self.task}'):
            os.makedirs(f'pdb_cache/{self.task}')
        for sequence in sequences:
            if not os.path.exists(f'pdb_cache/{self.task}/{sequence}.pdb'):
                output = self.model.infer_pdb(self.get_full_seq(sequence))
                with open(f"pdb_cache/{self.task}/{sequence}.pdb", "w") as f:
                    f.write(output)

    def calculate_rmsd(self, sequences):
        self.generate_pdb(sequences)

        scores = []
        parser = PDBParser(QUIET=True)
        superimposer = SVDSuperimposer()
        structure1 = parser.get_structure(self.wt_seg, f'pdb_cache/{self.task}/{self.wt_seg}.pdb')
        atoms1 = []
        for model in structure1:
            for chain in model:
                for i, residue in enumerate(chain):
                    try:
                        if i not in self.mut_pos:
                            continue
                        atoms1.append(residue['N'].get_coord())
                        atoms1.append(residue['CA'].get_coord())  
                        atoms1.append(residue['C'].get_coord())  
                    except KeyError:
                        continue

        for sequence in tqdm(sequences):
            structure2 = parser.get_structure(sequence, f'pdb_cache/{self.task}/{sequence}.pdb')
            atoms2 = []
            for model in structure2:
                for chain in model:
                    for i, residue in enumerate(chain):
                        try:
                            if i not in self.mut_pos:
                                continue
                            atoms2.append(residue['N'].get_coord())
                            atoms2.append(residue['CA'].get_coord())
                            atoms2.append(residue['C'].get_coord())
                        except KeyError:
                            continue

            superimposer.set(np.array(atoms1), np.array(atoms2))
            superimposer.run()

            # calculate rmsd on mutation sites
            atoms2_aligned = superimposer.get_transformed()
            sd = 0
            # for pos in self.mut_pos:
            for pos in [0,1,2,3]:
                sd += sum([(atoms1[pos*3][c] - atoms2_aligned[pos*3][c])**2 for c in range(3)])
                sd += sum([(atoms1[pos*3+1][c] - atoms2_aligned[pos*3+1][c])**2 for c in range(3)])
                sd += sum([(atoms1[pos*3+2][c] - atoms2_aligned[pos*3+2][c])**2 for c in range(3)])
            rmsd = (sd / (len(self.mut_pos) * 3))**0.5
            scores.append(rmsd)
        return scores

if __name__ == '__main__':
    import pandas as pd
    from scipy import stats
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['GB1', 'PhoQ'])
    parser.add_argument('--size', type=int, default=9999999)
    args = parser.parse_args()

    esmfold = ESMFold('GB1', 'MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE', [38, 39, 40,51], 'cuda')
    sequences = ['LMCG', 'LWCA', 'VYGV', 'FWAA']
    print(esmfold.calculate_rmsd(sequences))
