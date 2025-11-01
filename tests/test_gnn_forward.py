import pytest

torch = pytest.importorskip("torch", reason="torch not installed, skipping GNN forward test")
from me_drugban.gnn_backbone.gnn import SimpleGNN
import torch

def test_simple_gnn_forward():
    model = SimpleGNN(in_feats=10, hidden=8, out_feats=3)
    x = torch.randn(4, 10)
    out = model(x)
    assert out.shape == (4, 3)