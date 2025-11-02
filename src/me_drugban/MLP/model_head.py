import torch
import torch.nn as nn
import torch.nn.functional as F
# Map the fused features to the final prediction (binary, regression, etc.).
# Reference: Original DrugBAN: models.py lines 113-132
class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
    
# The GNN: shape (batch, n_drug_nodes, drug_feat_dim)
# The Protein CNN: shape (batch, n_prot_residues, prot_feat_dim)
# So,
# MLP in_dim = (n_drug_nodes * drug_feat_dim) + (n_prot_residues * prot_feat_dim)
# n_drug_nodes: number of nodes per molecule (after batching; usually max graph nodes in batch, or for the first molecule if no padding).
# drug_feat_dim: output feature dim from GNN (should match your hidden_feats[-1])
# n_prot_residues: length of output after CNN (depends on input protein length and kernel sizes/padding/stride)
# prot_feat_dim: number of output channels after last CNN layer