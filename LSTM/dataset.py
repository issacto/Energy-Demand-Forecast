import csv
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import random
from sklearn.preprocessing import MinMaxScaler


class SimpleDataset(Dataset):

    def __init__(self, path_to_csv, transform=None):
        colnames = ['date', 'value']
        df = pd.read_csv(path_to_csv, names=colnames, header=0)
        self.data = df.value.to_numpy()
        self.transform = transform

    def __len__(self):
        """ Returns the length of the dataset. """
        return self.data.shape[0]

    def __getitem__(self, index):
        """
        Returns one sample from the dataset, for a given index.
        """
        return self.data[index]


def normalize(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1))
    return torch.FloatTensor(data_normalized)


def inverse_transform(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.inverse_transform(np.array(data.reshape(-1, 1)))


def create_inout_sequences(input_data, train_window):
    inout_seq = []
    L = len(input_data)
    for i in range(L-train_window):
        train_seq = input_data[i:i+train_window]
        train_label = input_data[i+train_window:i+train_window+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def split_train_test(data, train_test_split):
    split = int(len(data)*train_test_split[0])
    train_set = data[:split]
    test_set = data[split:]
    return train_set, test_set
