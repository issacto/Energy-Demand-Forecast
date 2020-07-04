import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import dataset
import matplotlib.pyplot as plt


class LSTM(nn.Module):

    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
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
    loss = np.absolute(np.absolute(target-output) / target)
    return np.mean(loss)*100


if __name__ == '__main__':

    # Configure these
    path_to_csv = 'datasets-48149-87794-PJM_Load_hourly.csv'
    train_test_split = [0.8, 0.2]
    train_window = 10
    lr = 0.001
    loss_fn = nn.MSELoss()
    epochs = 1
    hidden_layer_size = 5

    input_set = dataset.SimpleDataset(path_to_csv)
    train_set = dataset.split_train_test(input_set, train_test_split)[0]
    test_set = dataset.split_train_test(input_set, train_test_split)[1]
    train_inout_seq = dataset.create_inout_sequences(
        dataset.normalize(train_set), train_window)

    model = LSTM(hidden_layer_size=hidden_layer_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_fn(y_pred, labels)
            print(str(single_loss))
            single_loss.backward()
            optimizer.step()
            print('at i = ' + str(i))

        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

fut_pred = len(test_set) - train_window - 1
test_inputs = dataset.normalize(test_set)[-train_window:].tolist()[0]
# print(test_inputs)

model.eval()

for i in range(fut_pred):

    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        # append predicted value to be used for predicting the next value
        test_inputs.append(model(seq).item())

print(type(test_set))
actual_predictions = dataset.inverse_transform(
    np.array(test_inputs).reshape(-1, 1), np.array(test_set[train_window:]).reshape(-1, 1))
print(actual_predictions)

"""Uncomment below for the percent error across the eval set"""
percentError = percent_error(
    np.array(actual_predictions), test_set[train_window:])
print(percentError)
print(np.array(actual_predictions)[6000:])
# print(test_set[train_window+201:train_window+221])

"""Plotting"""
x = np.arange(1, 6581-train_window, 1)
# x = np.arange(1, 1001, 1)
plt.title('Prediction vs Actual by Test Index')
plt.ylabel('Demand')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
# print("shape of " + str(np.expand_dims(
#     test_set[train_window:], axis=1).shape))
# print("shape of testset" + str(test_set[train_window:].shape))
# print("shape of pred" + str(actual_predictions.reshape(-1, 1).shape))
# plt.plot(np.expand_dims(x, axis=1), np.expand_dims(
#     test_set[train_window:], axis=1))
plt.plot(test_set[train_window:])
# plt.plot(np.expand_dims(x, axis=1), actual_predictions[:1000].reshape(-1, 1))
plt.plot(np.expand_dims(x, axis=1), actual_predictions.reshape(-1, 1))
plt.show()
