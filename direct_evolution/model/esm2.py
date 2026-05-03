import torch
import torch.nn as nn

from esm import pretrained

from ..utils.seq_utils import invalid_alphabet

class ESM(nn.Module):
    def __init__(self, args, wt_seq):
        super().__init__()
        self.args = args

        self.model = None
        self.alphabet = None
        self.device = args.device

        if args.esm == "esm2_t6_8M_UR50D":
            esm_model = pretrained.esm2_t6_8M_UR50D()
            self.embed_dim = 320
        if args.esm == "esm2_t12_35M_UR50D":
            esm_model = pretrained.esm2_t12_35M_UR50D()
            self.embed_dim = 480
        if args.esm == "esm2_t30_150M_UR50D":
            esm_model = pretrained.esm2_t30_150M_UR50D()
            self.embed_dim = 640
        if args.esm == "esm2_t33_650M_UR50D":
            esm_model = pretrained.esm2_t33_650M_UR50D()
            self.embed_dim = 1280
        
        self.esm_layer = int(args.esm.split("_")[1][1:])

        self.model = esm_model[0]
        self.alphabet = esm_model[1]
        
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.to(args.device)
        self.wt_seq = wt_seq
        
        labels, strs, self.wt_tokens = self.batch_converter([("wt", self.wt_seq)])
        self.wt_score = self.score(self.wt_tokens.to(self.device))
        self.wt_onehot = self.index_to_onehot(self.wt_tokens).to(self.device)
        self.wt_onehot = self.wt_onehot[0]
    
    def index_to_onehot(self, index):
        '''
            index: batch_size x seq_len
        '''
        one_hot = torch.nn.functional.one_hot(index.long(), num_classes=len(self.alphabet.all_toks))
        return one_hot
        
    def onehot_to_seq(self, x, vocab):
        '''
            x: bsz x seq_len x vocab_sz, tensor
            return: seq removing '<cls>', '<eos>', and rare amino acids
        '''
        toks = self.alphabet.all_toks
        seq_tks_idxs = torch.argmax(x, dim=2)
        
        result = []
        for seq_tks_idx in seq_tks_idxs:
            invalid = False
            seq = []
            for tk_idx in seq_tks_idx[1:-1]:
                tk = toks[tk_idx].upper()
                if tk not in vocab and tk in invalid_alphabet:     # ignore invalid seq ( )
                    invalid = True
                    break
                seq.append(tk)
            if invalid:
                continue
            else:
                result.append(''.join(seq))

        return result

    def forward(self, x, onehot=True):
        '''
            x: batch_size x (label, seq)
        '''
        if not onehot:
            labels, strs, tokens = self.batch_converter(x)
            tokens = tokens.to(self.device)
            result = self.model(tokens, repr_layers=[self.esm_layer])
            return result, labels
        else:
            result = self.model(x, repr_layers=[self.esm_layer])
            return result

    def score(self, x, delta=False):
        '''
            calculate esm output energy score

            intput:
                x: batch_size x seq_len

            output:
                score: logits score sum for each sample
        '''
        with torch.cuda.amp.autocast():
            outputs = self.model(x) 
            logits = outputs['logits']
            score = (self.index_to_onehot(x) * torch.nn.functional.log_softmax(logits, -1)).sum(dim=[1,2])
        if delta:
            return -score - self.wt_score.to(x.device)
        else:
            return -score
    
    def get_energy(self, x):
        energies = []
        for bi in range(0, x.size()[0], self.args.batch_size):
            energy = self.score(x[bi:bi+self.args.batch_size], delta=False)
            energies.append(energy)
        return torch.stack(energies)
    
    def get_energy_and_grad(self, x):
        x = x.requires_grad_()
        energy = self.score(x, delta=False)
        grad = torch.autograd.grad(energy.sum(), x)[0]
        return energy, grad
    
    