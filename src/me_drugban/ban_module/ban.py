import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFusion(nn.Module):
    def __init__(self):
        super(SimpleFusion, self).__init__()
    def forward(self, v_d, v_p):
        v_d_flat = v_d.view(v_d.size(0), -1)
        v_p_flat = v_p.view(v_p.size(0), -1)
        f = torch.cat([v_d_flat, v_p_flat], dim=1)
        return f
    
class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out):
        """Bilinear Attention Network (BAN) Layer for DrugBAN."""
        super(BANLayer, self).__init__()
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        # Linear projections
        self.v_proj = nn.Linear(v_dim, h_dim * h_out)
        self.q_proj = nn.Linear(q_dim, h_dim * h_out)

        # Attention
        self.att_proj = nn.Linear(h_dim, 1)

        # Output projection
        self.out_proj = nn.Linear(h_out * h_dim, h_dim)

    def forward(self, v, q):
        """
        v: (batch, num_drug_nodes, v_dim)
        q: (batch, num_prot_residues, q_dim)
        Returns:
            fused: (batch, h_dim)
            att:   (batch, h_out, num_drug_nodes, num_prot_residues)
        """
        batch_size, num_v, _ = v.size()
        _, num_q, _ = q.size()

        # Linear projections and reshape for heads
        v_ = self.v_proj(v)  # (batch, num_v, h_dim * h_out)
        q_ = self.q_proj(q)  # (batch, num_q, h_dim * h_out)
        v_ = v_.view(batch_size, num_v, self.h_out, self.h_dim)
        q_ = q_.view(batch_size, num_q, self.h_out, self.h_dim)

        # Compute bilinear interaction for each head
        v_ = v_.unsqueeze(2)  # (batch, num_v, 1, h_out, h_dim)
        q_ = q_.unsqueeze(1)  # (batch, 1, num_q, h_out, h_dim)
        joint = v_ * q_       # (batch, num_v, num_q, h_out, h_dim)

        # Move h_out to batch dimension for attention
        joint = joint.permute(0, 3, 1, 2, 4)  # (batch, h_out, num_v, num_q, h_dim)
        joint_flat = joint.reshape(batch_size * self.h_out, num_v, num_q, self.h_dim)

        # Compute attention over pairs (drug node x protein residue)
        att = self.att_proj(joint_flat)           # (batch*h_out, num_v, num_q, 1)
        att = att.squeeze(-1)                     # (batch*h_out, num_v, num_q)
        att = att.view(batch_size, self.h_out, num_v, num_q)
        att = F.softmax(att, dim=2)               # Softmax over drug nodes

        # Weighted sum to get head features
        joint = joint.view(batch_size, self.h_out, num_v * num_q, self.h_dim)
        att = att.view(batch_size, self.h_out, num_v * num_q).unsqueeze(-1)
        head_feat = (joint * att).sum(dim=2)      # (batch, h_out, h_dim)

        # Concatenate all heads
        head_feat = head_feat.view(batch_size, self.h_out * self.h_dim)
        fused = self.out_proj(head_feat)          # (batch, h_dim)

        return fused, att