from torch.utils.data import Dataset

class DomainAdaptationDTIDataset(Dataset):
    """
    Combines source and target datasets for cross-domain training.
    Adds a 'domain' label: 0=source, 1=target.
    """
    def __init__(self, source_dataset, target_dataset, mode='concat'):
        self.source = source_dataset
        self.target = target_dataset
        self.mode = mode
        if self.mode == 'concat':
            self.length = len(self.source) + len(self.target)
        elif self.mode == 'paired':
            self.length = min(len(self.source), len(self.target))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == 'concat':
            if idx < len(self.source):
                item = self.source[idx]
                item['domain'] = 0
                return item
            else:
                item = self.target[idx - len(self.source)]
                item['domain'] = 1
                return item
        elif self.mode == 'paired':
            s = self.source[idx]
            t = self.target[idx]
            s['domain'] = 0
            t['domain'] = 1
            return (s, t)