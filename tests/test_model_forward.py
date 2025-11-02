import torch
from me_drugban.model import DrugBAN
from me_drugban.data_loader.dataloader import MoleculeDataset
from me_drugban.data_loader.collate import collate_fn
from torch.utils.data import DataLoader

def test_model_forward():
    # Path to a small or test split CSV
    test_path = "data/bindingdb/random/test.csv"
    test_set = MoleculeDataset(test_path)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Model params (should match your config)
    drug_gnn_params = dict(in_feats=74, dim_embedding=128, hidden_feats=[128, 128, 128])
    protein_params = dict(embedding_dim=128, num_filters=[32, 64, 96], kernel_size=[7, 7, 7])
    ban_params = dict(v_dim=128, q_dim=96, h_dim=256, h_out=4)
    mlp_params = dict(in_dim=256, hidden_dim=256, out_dim=1, binary=1)
    model = DrugBAN(drug_gnn_params, protein_params, mlp_params, ban_params)

    batch = next(iter(test_loader))
    out = model(batch["graph"], batch["protein"])
    assert out.shape[0] == batch["label"].shape[0], \
        f"Output shape {out.shape} does not match label shape {batch['label'].shape}"
    print("Forward pass test successful! Output shape:", out.shape)

if __name__ == "__main__":
    test_model_forward()