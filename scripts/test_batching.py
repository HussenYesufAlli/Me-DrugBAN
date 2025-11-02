from torch.utils.data import DataLoader
from me_drugban.data_loader.dataloader import MoleculeDataset
from me_drugban.data_loader.collate import collate_fn

dset = MoleculeDataset("data/bindingdb_sample/train.csv")
loader = DataLoader(dset, batch_size=8, collate_fn=collate_fn)
batch = next(iter(loader))
print("Batched graph:", batch["graph"])
print("Batch size (proteins):", batch["protein"].shape)
print("Batch size (labels):", batch["label"].shape)