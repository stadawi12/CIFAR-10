from torch.utils.data import Dataset
from preprocess_data import load_data

class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels):
        self.images = data
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

if __name__ == "__main__":
    BATCH_NUMBER = 4
    data, labels = load_data(BATCH_NUMBER)
    dataset = CIFAR10Dataset(data, labels)
    print(len(dataset))