import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")

logger = logging.getLogger('psi')

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, q, k, v):
        weight = F.softmax(q.matmul(k.mT), dim=-1) / np.sqrt(self.sqrt_dim)
        return weight.matmul(v)

class AttentionLayer(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim)
        nn.init.normal_(self.linear)

    def forward(self, k, v):
        return F.softmax(F.relu(self.linear(k)), dim=1) * v

class AttnEncoder(nn.Module):
    def __init__(self, dim, dropout=0.1) -> None:
        super().__init__()

        self.attention = ScaledDotProductAttention(dim)
        self.linear_1 = nn.Linear(dim, dim)
        self.linear_2 = nn.Linear(dim, dim)
        self.linear_3 = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k, v):
        res = v
        v = self.attention(k, k, v)
        v = self.dropout(self.linear_3(F.relu(self.linear_2(F.relu(self.linear_1(v))))))
        v += res
        output = self.layer_norm(v)
        return output

class StructEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_size, embed_dim)))
        nn.init.normal_(self.embedding)
        self.emb_ln = nn.LayerNorm(embed_dim)
        self.emb_linear = nn.Linear(embed_dim, hidden_dim)
        self.encoders = nn.ModuleList([AttnEncoder(hidden_dim) for i in range(3)])

    def forward(self, x):
        '''
            x : one-hot
        '''
        x = self.emb_linear(self.emb_ln(x.matmul(self.embedding)))
        for enc_layer in self.encoders:
            x = enc_layer(x, x) 
        return x

class StructDecoder(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        out = self.mlp(x.sum(dim=1)/x.size(1)).squeeze(1)
        return out

class TargetDecoder(nn.Module):
    def __init__(self, hidden_dim, kernel_size) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(hidden_dim, 32, kernel_size, padding='valid')
        self.conv_2 = nn.Conv1d(32, 32, kernel_size, padding='same')
        self.conv_3 = nn.Conv1d(32, 32, kernel_size, padding='same')
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = x.permute(0,2,1)
        x = torch.relu(self.conv_1(x))
        x = torch.relu(self.conv_2(x))
        x = torch.relu(self.conv_3(x))
        out = self.mlp(torch.squeeze(self.global_max_pool(x), dim=-1)).squeeze(1)
        return out

class PSI(nn.Module):
    def __init__(self, args, vocab_size, embed_dim, hidden_dim, seq_len):
        super().__init__()
        self.args = args
        self.device = args.device
        self.vocab_size = vocab_size
        self.struct_encoder = StructEncoder(vocab_size, embed_dim, hidden_dim)
        self.struct_decoder = TargetDecoder(hidden_dim, kernel_size=3)
        self.fitness_decoder = TargetDecoder(hidden_dim, kernel_size=3)

    def encode(self, x, train=True):
        self.struct_encoder.train(train)
        y = self.struct_encoder(x)
        return y
    
    def struct_decode(self, y):
        y = self.struct_decoder(y)
        return y
    
    def fitness_decode(self, y):
        y = self.fitness_decoder(y)
        return y

    def struct_forward(self, x):
        self.struct_encoder.train()
        x = self.struct_encoder(x)
        y = self.struct_decoder(x)
        return y
    
    def forward(self, x, encoder_train=False):
        # with torch.no_grad():
        self.struct_encoder.train(encoder_train)
        x = self.struct_encoder(x)
        y = self.fitness_decoder(x)
        return y

    def get_energy(self, x):
        energy = -torch.log(torch.sigmoid(self.forward(x)))
        return energy

    def get_energy_and_grad(self, x):
        x = x.requires_grad_()
        energy = -torch.log(torch.sigmoid(self.forward(x)))
        grad = torch.autograd.grad(energy.sum(), x, allow_unused=True)[0]
        return energy, grad