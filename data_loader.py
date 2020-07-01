import csv
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch


class SimpleDataset(Dataset):

    def __init__(self, path_to_csv, transform=None):
        df = pd.read_csv(path_to_csv)
        raw_data = df.to_numpy()
        transformed_data = transform(raw_data)
        self.data = transformed_data
        self.num_features = self.data.shape[1]-1
        self.features = self.data[:, :-1]
        self.labels = self.data[:, -1]
        # self.transform = transform

    def __len__(self):
        """ Returns the length of the dataset. """
        return self.data.shape[0]

    def __getitem__(self, index):
        """ 
        Returns one sample from the dataset, for a given index. 
        type index: int
        rtype: torch
        """
        sample = self.data[index]
        x = torch.from_numpy(np.array(sample[:-1]))
        y = torch.from_numpy(np.array(sample[-1]))
        return x, y


def get_data_loader(path_to_csv, train_test_split, transform_fn=None, batch_size=32):
    """
    """
    dataset = SimpleDataset(path_to_csv, transform=transform_fn)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    DataLoader
