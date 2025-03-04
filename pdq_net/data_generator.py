import os
import torch
from torch.utils.data import Dataset, DataLoader
import config


class SpikeDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)
        self.spike_trains = data['spike_trains']
        self.labels = data['labels']

    def __len__(self):
        return len(self.spike_trains)

    def __getitem__(self, idx):
        return self.spike_trains[idx], self.labels[idx]


def get_data_loader(data_path, batch_size=config.BATCH_SIZE, shuffle=True):
    dataset = SpikeDataset(data_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


if __name__ == "__main__":
    train_data_path = os.path.join(config.PROCESSED_DATA_DIR, 'train_data.pt')
    test_data_path = os.path.join(config.PROCESSED_DATA_DIR, 'test_data.pt')

    train_loader = get_data_loader(train_data_path, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = get_data_loader(test_data_path, batch_size=config.BATCH_SIZE, shuffle=False)

    for spike_trains, labels in train_loader:
        print(spike_trains.size(), labels.size())
        break
