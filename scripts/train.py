import torch
from torch.utils.data import DataLoader
from me_drugban.data_loader.dataloader import MoleculeDataset
from me_drugban.data_loader.collate import collate_fn
from me_drugban.model import DrugBAN
from me_drugban.train_loop.train import train

# 1. Hyperparameters and paths
epochs = 20
batch_size = 32
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Model and dataset parameters (edit as needed)
drug_gnn_params = dict(in_feats=74, dim_embedding=128, hidden_feats=[128, 128, 128])
protein_params = dict(embedding_dim=128, num_filters=[32, 64, 96], kernel_size=[7, 7, 7])
ban_params = dict(v_dim=128, q_dim=96, h_dim=256, h_out=4)
mlp_params = dict(in_dim=256, hidden_dim=256, out_dim=1, binary=1)

# Paths to your csv files
train_path = "data/bindingdb/random/train.csv"
val_path   = "data/bindingdb/random/val.csv"
test_path  = "data/bindingdb/random/test.csv"

# 3. Prepare datasets & loaders
train_set = MoleculeDataset(train_path)
val_set = MoleculeDataset(val_path)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 4. Model, optimizer, criterion
model = DrugBAN(drug_gnn_params, protein_params, mlp_params, ban_params).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.BCEWithLogitsLoss()

# 5. Train
save_path = "best_drugban.pt"
history = train(model, train_loader, val_loader, epochs, optimizer, criterion, device, save_path=save_path)

# 6. (Optional) Save training history/results
import pickle
with open("training_history.pkl", "wb") as f:
    pickle.dump(history, f)

print("Training complete. Best model saved to", save_path)