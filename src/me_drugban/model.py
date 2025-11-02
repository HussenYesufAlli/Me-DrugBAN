import torch
import torch.nn as nn
from .gnn_backbone.gnn import MolecularGCN
from .protein_encoder.encoder import ProteinCNN
from .ban_module.ban import BANLayer   # <-- Use BANLayer!
from .MLP.model_head import MLPDecoder

class DrugBAN(nn.Module):
    def __init__(self, drug_gnn_params, protein_params, mlp_params, ban_params):
        super(DrugBAN, self).__init__()
        self.drug_extractor = MolecularGCN(**drug_gnn_params)
        self.protein_extractor = ProteinCNN(**protein_params)
        self.fusion = BANLayer(**ban_params)
        self.mlp_classifier = MLPDecoder(**mlp_params)

    def forward(self, batch_graph, prot_tensor):
        v_d = self.drug_extractor(batch_graph)
        v_p = self.protein_extractor(prot_tensor)
        f, att = self.fusion(v_d, v_p)   # BANLayer returns (fused_feature, attention_map)
        score = self.mlp_classifier(f)
        return score