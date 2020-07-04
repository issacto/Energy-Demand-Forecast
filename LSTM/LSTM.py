import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import dataset
import matplotlib.pyplot as plt

#expected loss < 1%

class LSTM(nn.Module):

    def __init__(self, input_size=1, hidden_layer_size=5, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, x):
        lstm_out, self.hidden_cell = self.lstm(
            x.view(len(x), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(x), -1))
        return predictions[-1]


def percent_error(output, target):
    loss = torch.abs(torch.div(torch.abs(target-output), target))
    return torch.mean(loss)*100


if __name__ == '__main__':

    # Configure these
    path_to_csv = 'datasets-48149-87794-PJM_Load_hourly.csv'
    train_test_split = [0.8, 0.2]
    train_window = 365
    lr = 0.001
    loss_fn = nn.MSELoss()
    epochs = 1250

    input_set = dataset.SimpleDataset(path_to_csv)
    train_set = dataset.split_train_test(input_set, train_test_split)[0]
    test_set = dataset.split_train_test(input_set, train_test_split)[1]
    train_inout_seq = dataset.create_inout_sequences(
        dataset.normalize(train_set), train_window)

    model = LSTM()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_fn(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

fut_pred = 12
test_inputs = dataset.normalize(test_set)[-train_window:].tolist()
print(test_inputs)

model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())

actual_predictions = dataset.inverse_transform(test_set[train_window:])
print(actual_predictions)

"""Uncomment below for the percent error across the eval set"""

# percentError = percent_error(preds, y.view(1, len(y)))
# print(percentError)

"""Plotting"""
x = np.arange(132, 144, 1)
print(x)
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)

plt.plot(input_set[-train_window:])
plt.plot(x, actual_predictions)
plt.show()
