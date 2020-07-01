import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import data_loader as dl


class Model(nn.Module):

    def __init__(self, num_parem):
        super(Model, self).__init__()
        self.num_param = num_param
        self.thetas = torch.nn.Parameter(torch.randn(1, self.num_param))

    def forward(self, x):
        return torch.mm(self.thetas, torch.t(x).float())


def data_transform(sample):
    return np.append([1], sample)


def percent_error(output, target):
    loss = torch.abs(torch.div(torch.abs(target-output), target))
    return torch.mean(loss)*100


if __name__ == '__main__':

    # Configure these
    path_to_csv = 'window.csv'
    transform_fn = None
    train_test_split = [0.7, 0.3]
    batch_size = 32
    lr = 0.01
    loss_fn = nn.MSELoss()
    num_param = 3  # can be obtained from loaders
    TOTAL_TIME_STEPS = 100

    train_loader, test_loader =\
        dl.get_data_loaders(
            path_to_csv,
            transform_fn=transform_fn,
            train_test_split=train_test_split,
            batch_size=batch_size)

    model = Model(num_param)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()
    for t in range(TOTAL_TIME_STEPS):
        for batch_index, (input_t, y) in enumerate(train_loader):

            optimizer.zero_grad()
            print(model.thetas)
            preds = model(input_t)
            print('HELLLLOOOOOO')
            print(preds)
            print(y)
            # You might have to change the shape of things here.
            loss = loss_fn(preds, y)
            loss.backward()
            # print(loss)
            optimizer.step()

            model.eval()

    for batch_index, (input_t, y) in enumerate(test_loader):

        preds = model(input_t)

        print(preds)
        loss = loss_fn(preds, y.view(1, len(y)))

        """Uncomment below for the percent error across the eval set"""

        #percentError = percent_error(preds, y.view(1, len(y)))
        # print(percentError)

        """Function plot_linear is used for Dataset 1 because it has two features.
        Use function plot_linear_2D for Dataset 2."""

        # plot.plot_linear(preds, input_t, y.view(1, len(y)))
