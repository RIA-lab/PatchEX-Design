import torch
import torch.nn as nn

def safe_logits_to_probs(logits):
    """safe convert logits to probs"""
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits, dim=-1)
    return probs
