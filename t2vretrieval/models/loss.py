"""
Loss functions.
"""
from torch.nn import functional as F
import torch
from torch import nn

INF = 32752

def cosine_sim(im, s):
    im = F.normalize(im)
    s = F.normalize(s)
    sim = im.mm(s.t())
    return sim


class ContrastiveLoss(nn.Module):
    """
    Regular Contrastive Loss between 2 groups of embeddings
    """
    def __init__(self, margin = 0.2, max_violation = False, norm = True, topk = 1):
        super().__init__()
        self.margin = margin
        self.norm = norm
        self.max_violation = max_violation
        self.topk = topk
        self.sim = cosine_sim

    def forward(self, im, s, max_violation = True):
        """
        Inputs shape (batch, embed_dim)

        Args:
            im: Visual embeddings (batch, embed_dim)
            s: Text embeddings (batch, embed_dim)

        Returns:
        """
        # compute image-sentence score matrix - how close is im(y) to s(x)
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals, where there is just the margin left
        mask = torch.eye(scores.shape[0]).bool().to(scores.device)
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
            if self.norm:
                return (cost_s.sum() + cost_im.sum()).div(im.shape[0])
        else:
            if self.norm:
                return (cost_s.sum() + cost_im.sum()).div(im.shape[0] * (s.shape[0] - 1))

        return cost_s.sum() + cost_im.sum()