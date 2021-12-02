import torch as th
from torch import nn

class PositionalEncodingSinCos(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Modified: Removed embedding, changed the calculation. Should be fine.

    Args:
        dim: embedding size
        dropout_prob: dropout parameter
        max_len: Maximum input length.
    """

    def __init__(self, dim, dropout_prob = 0., max_len = 1000):
        super().__init__()
        pe = th.zeros(max_len, dim).float()
        position = th.arange(0, max_len).unsqueeze(1).float()
        dimension = th.arange(0, dim).float()
        div_term = 10000 ** (2 * dimension / dim)
        # print(div_term.shape)
        pe[:, 0::2] = th.sin(position / div_term[0::2])
        pe[:, 1::2] = th.cos(position / div_term[1::2])
        # print(f"Positional Encoding max_len x dim: {pe.shape}\n", pe, "\n",
        #       sep="")
        # div_term = torch.exp((torch.arange(0, dim, 2) *
        #                       -(math.log(10000.0) / dim)).float())
        # pe[:, 0::2] = torch.sin(position.float() * div_term)
        # pe[:, 1::2] = torch.cos(position.float() * div_term)
        # pe = pe.unsqueeze(0)

        # put it into state dict even though it is not learnable
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim

    def forward(self, x, step=None):
        # x *= math.sqrt(self.dim) # not sure
        # print(x.size(1))
        assert step is None, "Never used step"
        x = x + self.pe[:x.shape[1], :]
        # else:
        #     # x = x + self.pe[:, step]
        x = self.dropout(x)
        return x
