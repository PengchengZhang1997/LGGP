import torch as th
from torch import nn

class LayerNormalization(nn.Module):
    """
    Layer Normalization - Normalize across features instead of across the
    batch like in BatchNorm. Independent of batch size.

    Different results from the PyTorch implementation.
    """
    def __init__(self, normalized_shape, epsilon = 1e-6):
        super().__init__()

        self.gain = nn.Parameter(th.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(th.zeros(normalized_shape), requires_grad=True)
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gain * (x - mean) / (std + self.epsilon) + self.bias
