from torch.utils.data import Dataset

class NomenclatureDataset(Dataset):
    def __init__(self, chemical_names):
        self.names = chemical_names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.names[idx]
