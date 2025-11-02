import torch
import torch.nn as nn
from dgllife.model.gnn import GCN
import dgl
# Encode each molecule (DGLGraph with node features) into a learnable representation using a Graph Convolutional Network (GCN).
# Reference: Original DrugBAN: models.py lines 74-90

class MolecularGCN(nn.Module):
    def __init__(self, in_feats=74, dim_embedding=128, hidden_feats=[128,128,128], activation=None, padding=True):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata['h']
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        # Store processed node features back in the graph under a new key
        batch_graph.ndata['gnn_out'] = node_feats
        graph_embeds = dgl.mean_nodes(batch_graph, 'gnn_out')
        return graph_embeds