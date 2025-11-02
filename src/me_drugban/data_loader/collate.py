import torch
import dgl

def collate_fn(batch):
    graphs = [item["graph"] for item in batch]
    prot_seqs = [item["protein_encoded"] for item in batch]
    labels = [item["label"] for item in batch]
    batched_graph = dgl.batch(graphs)
    prot_tensor = torch.tensor(prot_seqs, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    return {
        "graph": batched_graph,
        "protein": prot_tensor,
        "label": label_tensor
    }