from torch.utils.data import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels):
        self.images = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

