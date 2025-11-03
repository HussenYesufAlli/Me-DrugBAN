import pandas as pd
from torch.utils.data import Dataset

class CSVDrugTargetDataset(Dataset):
    """
    General CSV-based DTI dataset loader for DAVIS/KIBA/BindingDB.
    Expects columns in CSV such as: Drug, Target, Y.
    Optionally supports pre-defined split files containing Drug and Target columns.
    """
    def __init__(self, dataset_csv, subset_csv=None, drug_col="Drug", target_col="Target", affinity_col="Y"):
        """
        Args:
            dataset_csv: Path to the main dataset CSV (e.g., 'doc/dataset/DAVIS.csv').
            subset_csv: Path to the split CSV (e.g., 'doc/split/DAVIS_train.csv').
            drug_col: Name of the drug/SMILES column in the dataset.
            target_col: Name of the protein/sequence column in the dataset.
            affinity_col: Name of the affinity/label column in the dataset.
        """
        self.data = pd.read_csv(dataset_csv)
        self.drug_col = drug_col
        self.target_col = target_col
        self.affinity_col = affinity_col

        if subset_csv is not None:
            split = pd.read_csv(subset_csv)
            # Keep only Drug and Target columns for merging
            split = split[[drug_col, target_col]].drop_duplicates()
            self.data = self.data.merge(split, on=[drug_col, target_col], how='inner')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            "drug": row[self.drug_col],
            "target": row[self.target_col],
            "affinity": row[self.affinity_col]
        }