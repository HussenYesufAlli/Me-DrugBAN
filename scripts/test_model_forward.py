import torch
from me_drugban.model import DrugBAN
from me_drugban.data_loader.dataloader import MoleculeDataset
from me_drugban.data_loader.collate import collate_fn

# --- Parameter setup for BANLayer ---
drug_gnn_params = dict(in_feats=74, dim_embedding=128, hidden_feats=[128, 128, 128])
protein_params = dict(embedding_dim=128, num_filters=[32, 64, 96], kernel_size=[7, 7, 7])

ban_params = dict(
    v_dim=128,   # Output dim of GNN
    q_dim=96,    # Output dim of last CNN layer
    h_dim=256,   # BAN hidden dim (matches MLP in_dim)
    h_out=4      # Number of attention heads
)

mlp_params = dict(
    in_dim=256,   # matches ban_params['h_dim']
    hidden_dim=256,
    out_dim=128,
    binary=1
)

# --- Instantiate the model with BANLayer ---
model = DrugBAN(drug_gnn_params, protein_params, mlp_params, ban_params)

dset = MoleculeDataset("data/bindingdb_sample/train.csv")
from torch.utils.data import DataLoader
loader = DataLoader(dset, batch_size=2, collate_fn=collate_fn)
batch = next(iter(loader))

out = model(batch["graph"], batch["protein"])
print("Model output shape:", out.shape)