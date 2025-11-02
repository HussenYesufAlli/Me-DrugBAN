# import pandas as pd

# class MoleculeDataset:
#     def __init__(self, csv_path):
#         self.df = pd.read_csv(csv_path)
#         # Optionally validate columns
#         expected = {"SMILES", "Protein", "Y"}
#         if not expected.issubset(self.df.columns):
#             raise ValueError(f"CSV must have columns {expected}, got {self.df.columns}")

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         return {
#             "smiles": row["SMILES"],
#             "protein": row["Protein"],
#             "label": row["Y"]
#         }


import pandas as pd
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from functools import partial
from .utils import integer_encode_protein  # Make sure this is implemented!

class MoleculeDataset:
    def __init__(self, csv_path, max_drug_nodes=290):
        self.df = pd.read_csv(csv_path)
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.smiles_to_graph = partial(smiles_to_bigraph, add_self_loop=True)
        self.max_drug_nodes = max_drug_nodes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row["SMILES"]
        graph = self.smiles_to_graph(
            smiles=smiles,
            node_featurizer=self.atom_featurizer,
            edge_featurizer=self.bond_featurizer
        )
        protein_seq = row["Protein"]
        protein_encoded = integer_encode_protein(protein_seq)
        return {
            "graph": graph,
            "smiles": smiles,
            "protein": protein_seq,
            "protein_encoded": protein_encoded,
            "label": row["Y"]
        }