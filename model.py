import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import data_loader as dl
import plotting as plot


class NeutralNetwork(nn.Module):

    def __init__(self, num_parem):
        super().__init__()
        self.hidden1 = Linear()

    def forward(self, x):

    def windowing():

    def percent_error(output, target):

    if __name__ == '__main__':

        # Configure these
        path_to_csv = 'datasets-48149-87794-PJM_Load_hourly.csv'
        transform_fn = windowing()
        train_test_split = [0.7, 0.3]
        batch_size = 32
        lr = 0.01
        loss_fn = ?
        num_param = ?
        TOTAL_TIME_STEPS = 100

        train_loader, test_loader =\
            dl.get_data_loaders(
                path_to_csv,
                transform_fn=transform_fn,
                train_test_split=train_test_split,
                batch_size=batch_size)
