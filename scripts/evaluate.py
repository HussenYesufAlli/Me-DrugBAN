import torch
from torch.utils.data import DataLoader
from me_drugban.data_loader.dataloader import MoleculeDataset
from me_drugban.data_loader.collate import collate_fn
from me_drugban.model import DrugBAN
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
import pandas as pd
import os

# ---- Configuration ----
test_path = "data/bindingdb/random/test.csv"
model_path = "best_drugban.pt"
batch_size = 32

# Model hyperparameters (should match training config)
drug_gnn_params = dict(in_feats=74, dim_embedding=128, hidden_feats=[128, 128, 128])
protein_params = dict(embedding_dim=128, num_filters=[32, 64, 96], kernel_size=[7, 7, 7])
ban_params = dict(v_dim=128, q_dim=96, h_dim=256, h_out=4)
mlp_params = dict(in_dim=256, hidden_dim=256, out_dim=1, binary=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Prepare test data ----
test_set = MoleculeDataset(test_path)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ---- Load model ----
model = DrugBAN(drug_gnn_params, protein_params, mlp_params, ban_params).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for batch in test_loader:
        graphs = batch["graph"].to(device)
        proteins = batch["protein"].to(device)
        labels = batch["label"].to(device)
        logits = model(graphs, proteins)
        probs = torch.sigmoid(logits).squeeze(-1)
        preds = (probs >= 0.5).int()
        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

# ---- Concatenate and compute metrics ----
all_labels = torch.cat(all_labels).numpy()
all_preds = torch.cat(all_preds).numpy()
all_probs = torch.cat(all_probs).numpy()

acc = accuracy_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
conf_mat = confusion_matrix(all_labels, all_preds)

print(f"Test Accuracy:  {acc:.4f}")
print(f"Test ROC-AUC:   {auc:.4f}")
print(f"F1 Score:       {f1:.4f}")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print("Confusion Matrix:")
print(conf_mat)

# ---- Export predictions to CSV ----
df = pd.read_csv(test_path)
df["label"] = all_labels
df["probability"] = all_probs
df["prediction"] = all_preds

os.makedirs("results", exist_ok=True)
df.to_csv("results/test_predictions.csv", index=False)
print("Predictions exported to results/test_predictions.csv")