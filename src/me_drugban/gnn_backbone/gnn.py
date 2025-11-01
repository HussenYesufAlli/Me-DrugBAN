"""A minimal GNN-like backbone (placeholder) using torch.nn.

This is intentionally simple: a 2-layer MLP that can be replaced by
a full DGL/torch-geometric GNN later.
"""

try:
    import torch
    import torch.nn as nn
except Exception:  # keep module importable even if torch missing
    torch = None
    nn = None

class SimpleGNN(nn.Module if nn is not None else object):
    def __init__(self, in_feats: int = 10, hidden: int = 32, out_feats: int = 2):
        if nn is None:
            raise RuntimeError("torch is required to instantiate SimpleGNN")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_feats),
        )

    def forward(self, x):
        return self.net(x)