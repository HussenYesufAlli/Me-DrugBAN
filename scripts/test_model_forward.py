import torch
from me_drugban.model import DrugBAN
from me_drugban.data_loader.dataloader import MoleculeDataset
from me_drugban.data_loader.collate import collate_fn

# Use the actual fusion output shape for in_dim!
drug_gnn_params = dict(in_feats=74, dim_embedding=128, hidden_feats=[128, 128, 128])
protein_params = dict(embedding_dim=128, num_filters=[32, 64, 96], kernel_size=[7, 7, 7])
mlp_params = dict(in_dim=116800, hidden_dim=256, out_dim=128, binary=1)  # adjust in_dim if you use padding

model = DrugBAN(drug_gnn_params, protein_params, mlp_params)

dset = MoleculeDataset("data/bindingdb_sample/train.csv")
from torch.utils.data import DataLoader
loader = DataLoader(dset, batch_size=2, collate_fn=collate_fn)
batch = next(iter(loader))

out = model(batch["graph"], batch["protein"])
print("Model output shape:", out.shape)