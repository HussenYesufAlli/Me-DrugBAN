"""High-level BAN module that composes encoders and GNN backbones."""

from typing import Any

class BAN:
    """Simple composition class. Replace with training/inference logic."""

    def __init__(self, protein_encoder: Any, gnn_backbone: Any):
        self.protein_encoder = protein_encoder
        self.gnn_backbone = gnn_backbone

    def predict(self, protein_seq: str, mol_features):
        prot_vec = self.protein_encoder.encode(protein_seq)
        # mol_features is expected to be a list/array of floats
        return {"prot_vec_len": len(prot_vec), "mol_feats_len": len(mol_features)}
    