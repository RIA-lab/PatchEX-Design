from transformers import AutoTokenizer, EsmForProteinFolding, AutoModel
import torch
import json
from .structure import get_raw_accession, compute_tmscore
import os


class Evaluator:
    def __init__(self, task: str, batch_size: int = 16, reference_pdb_dir: str = "pdb"):
        """Initialize Hugging Face ESMFold model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.task = task
        self.batch_size = batch_size

        self.tokenizer_fold = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.esm_fold = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device)
        self.esm_fold.eval()

        self.reference_pdb_dir = reference_pdb_dir


    def __call__(self, result_path, reference_pdb=None):
        if isinstance(result_path, str):
            with open(result_path, "r") as f:
                res = json.load(f)
            output_dir = "/".join(result_path.split("/")[:-1])
        elif isinstance(result_path, dict):
            res = result_path
            output_dir = result_path.pop('output_dir')
        else:
            raise ValueError("result_path must be a str or dict")

        accession = res["accession"]
        raw_accession = get_raw_accession(accession)
        if not reference_pdb:
            reference_pdb = f"{self.reference_pdb_dir}/{self.task}/{raw_accession}.pdb"
        sequence = res["sequence"]
        # ec = res["ec"]


        if not os.path.exists(reference_pdb):
            raise FileNotFoundError(reference_pdb)

        # Tokenize input sequence
        inputs = self.tokenizer_fold([sequence], return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.esm_fold(**inputs)

        folded_positions = outputs.positions[-1, 0].detach().cpu().numpy()
        plddt = outputs.plddt[0].mean(dim=-1).detach().cpu().numpy()

        del outputs
        del inputs
        torch.cuda.empty_cache()

        # Save PDB file
        pdb_pred = f"{output_dir}/{accession}_pred.pdb"
        self.write_pdb(sequence, folded_positions, plddt, pdb_pred)

        mean_plddt = float(plddt.mean())

        # Compute structural similarity metrics
        tm, rmsd = compute_tmscore(pdb_pred, reference_pdb)

        metric = {
            "mean_plddt": mean_plddt,
            "tm_score": tm,
            "rmsd": rmsd,
            # "functional_embedding_similarity": cosine_sim
        }

        return metric




    # def get_esm_embedding(self, sequence):
    #     """Compute mean-pooled ESM-2 embedding for a sequence."""
    #     embeddings = []
    #
    #     for batch_start in range(0, len(sequence), self.batch_size):
    #         batch_seqs = sequence[batch_start: batch_start + self.batch_size]
    #
    #         with torch.no_grad():
    #             # Padding is crucial for batching sequences of different lengths
    #             tokens = self.tokenizer_esm(
    #                 batch_seqs,
    #                 return_tensors="pt",
    #                 padding=True,  # Add padding
    #                 add_special_tokens=True
    #             )
    #             tokens = {k: v.to(self.device) for k, v in tokens.items()}
    #             outputs = self.esm(**tokens)
    #
    #             # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_dim)
    #             # Mean pooling across sequence length, ignoring padding tokens
    #             attention_mask = tokens['attention_mask'].unsqueeze(-1)  # (batch, seq_len, 1)
    #             masked_embeddings = outputs.last_hidden_state * attention_mask
    #             sum_embeddings = masked_embeddings.sum(dim=1)  # (batch, hidden_dim)
    #             sum_mask = attention_mask.sum(dim=1)  # (batch, 1)
    #             mean_embeddings = sum_embeddings / sum_mask  # (batch, hidden_dim)
    #
    #             embeddings.append(mean_embeddings.cpu())
    #
    #     # Concatenate all batch embeddings
    #     return torch.cat(embeddings, dim=0)


    @staticmethod
    def load_fasta(fasta_path):
        """Load sequences from a FASTA file.

        Returns:
            dict: Dictionary mapping sequence IDs to sequences
        """
        sequences = {}
        current_id = None
        current_seq = []

        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    # Save previous sequence if it exists
                    if current_id is not None:
                        sequences[current_id] = "".join(current_seq)
                    # Start new sequence
                    current_id = line[1:]  # Remove ">" character
                    current_seq = []
                else:
                    # Add to current sequence
                    current_seq.append(line)

            # Don't forget the last sequence
            if current_id is not None:
                sequences[current_id] = "".join(current_seq)

        return sequences

    @staticmethod
    def write_fasta(output_path, sequences, line_width=60):
        """Write sequences to a FASTA file.

        Args:
            output_path (str): Path to output FASTA file
            sequences (dict or list): Either a dict mapping accessions to sequences,
                                      or a list of (accession, sequence) tuples
            line_width (int): Number of characters per line for sequence (default: 60)
                             Set to None for no line wrapping
        """
        with open(output_path, "w") as f:
            # Handle both dict and list inputs
            items = sequences.items() if isinstance(sequences, dict) else sequences

            for accession, sequence in items:
                # Write header line
                f.write(f">{accession}\n")

                # Write sequence with line wrapping
                if line_width is None:
                    f.write(f"{sequence}\n")
                else:
                    for i in range(0, len(sequence), line_width):
                        f.write(f"{sequence[i:i + line_width]}\n")

    @staticmethod
    def write_pdb(sequence, coords, plddt, out_path):
        """Write PDB file in strict PDB format for TMalign compatibility."""

        # Standard 3-letter amino acid codes
        aa_map = {
            'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
            'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
            'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
            'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
            'X': 'UNK'
        }

        # Backbone atoms with their element symbols
        backbone_atoms = [
            ("N", "N"),  # Nitrogen
            ("CA", "C"),  # Carbon alpha
            ("C", "C"),  # Carbon
            ("O", "O")  # Oxygen
        ]

        with open(out_path, "w") as f:
            atom_index = 1

            for res_i, (aa, atom_coords, b) in enumerate(zip(sequence, coords, plddt), start=1):
                # Convert single letter to 3-letter code
                aa_3letter = aa_map.get(aa.upper(), 'UNK')

                # Write backbone atoms
                for atom_j in range(min(4, len(atom_coords))):
                    atom_name, element = backbone_atoms[atom_j]
                    x, y, z = atom_coords[atom_j]

                    # Strict PDB format
                    line = (
                        f"ATOM  "  # 1-6
                        f"{atom_index:5d} "  # 7-11 + space
                        f" {atom_name:<3s}"  # 12-16
                        f" "  # 17
                        f"{aa_3letter:>3s} "  # 18-21
                        f"A"  # 22
                        f"{res_i:4d}"  # 23-26
                        f"    "  # 27-30
                        f"{x:8.3f}"  # 31-38
                        f"{y:8.3f}"  # 39-46
                        f"{z:8.3f}"  # 47-54
                        f"  1.00"  # 55-60
                        f"{b:6.2f}"  # 61-66
                        f"          "  # 67-76
                        f" {element:>1s}"  # 77-78 (correct element!)
                        f"\n"
                    )

                    f.write(line)
                    atom_index += 1

            f.write("END\n")
