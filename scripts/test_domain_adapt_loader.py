from me_drugban.data_loader.davis_kiba_loader import CSVDrugTargetDataset
from me_drugban.domain_adapt.adaptor import DomainAdaptationDTIDataset
from torch.utils.data import DataLoader

# All of KIBA as source, DAVIS train as target
kiba_train = CSVDrugTargetDataset(
    'doc/dataset/KIBA.csv',
    drug_col='Drug', target_col='Target', affinity_col='Y'
)
davis_train = CSVDrugTargetDataset(
    'doc/dataset/DAVIS.csv',
    subset_csv='doc/split/DAVIS_train.csv',
    drug_col='Drug', target_col='Target', affinity_col='Y'
)

domain_dataset = DomainAdaptationDTIDataset(kiba_train, davis_train)
loader = DataLoader(domain_dataset, batch_size=64, shuffle=True)

for batch in loader:
    print(batch['domain'])  # Should print 0 (KIBA) or 1 (DAVIS)
    break