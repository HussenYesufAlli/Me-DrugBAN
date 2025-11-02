import matplotlib.pyplot as plt
import networkx as nx
from me_drugban.data_loader.dataloader import MoleculeDataset

# Load one sample from your dataset
dset = MoleculeDataset("data/bindingdb_sample/train.csv")
sample = dset[0]
g = sample["graph"]

# Convert DGL graph to NetworkX for easy plotting
nx_g = g.to_networkx().to_undirected()

plt.figure(figsize=(6, 6))
nx.draw(nx_g, with_labels=True, node_color="skyblue", node_size=600)
plt.title("Molecule as Graph (nodes=atoms, edges=bonds)")
plt.show()