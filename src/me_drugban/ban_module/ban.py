import torch
import torch.nn as nn
    # Combine the drug (GNN) and protein (CNN) feature matrices into a joint representation.
    # Original DrugBAN uses a Bilinear Attention Network (BAN), which is implemented as a custom module.
class SimpleFusion(nn.Module):
    def __init__(self):
        super(SimpleFusion, self).__init__()
    def forward(self, v_d, v_p):
        # v_d: (batch, drug_seq, d_dim), v_p: (batch, prot_seq, p_dim)
        # For simple fusion, flatten and concatenate
        v_d_flat = v_d.view(v_d.size(0), -1)
        v_p_flat = v_p.view(v_p.size(0), -1)
        f = torch.cat([v_d_flat, v_p_flat], dim=1)
        return f
        