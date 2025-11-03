from me_drugban.data_loader.davis_kiba_loader import CSVDrugTargetDataset

# Full dataset usage example
dataset = CSVDrugTargetDataset('doc/dataset/DAVIS.csv')

print(f"Loaded {len(dataset)} samples from full DAVIS.")
print("Sample 0:", dataset[0])

# With split
trainset = CSVDrugTargetDataset('doc/dataset/DAVIS.csv', subset_csv='doc/split/DAVIS_train.csv')
print(f"Loaded {len(trainset)} samples from DAVIS_train.")
# trainset = CSVDrugTargetDataset('doc/dataset/KIBA.csv', subset_csv='doc/split/KIBA_train.csv')
# print(f"Loaded {len(trainset)} samples from KIBA_train")
print("Sample 0:", trainset[0])