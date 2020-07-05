# Energy-Demand-Forecast
Data provided by Kaggle
<br>
Package: Keras, Matlab
<br>
Members: Winnie Chow, Issac To


## Data Prepreation
```bash
dataset=np.loadtxt("windowWith10Inputs.csv", delimiter=",")
```
We first prepare the data by converting the raw data into a 10-input-wide Window via csvConverter.py.
<br>After that, we load the data and normalize it

## Neural Network
```python
model = Sequential()
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(1, activation='linear'))
```
We use Adam as optimizer and the early stopping's patience is set to be 3.
```python
es = EarlyStopping(monitor='val_loss',patience=3)
history = model.fit(X_train, y_train, epochs=20,validation_data=(X_test, y_test), batch_size=32,callbacks=[es])
```
## Output
<img src="https://github.com/issacto/Energy-Demand-Forecast/blob/master/Image/trainingvstestloss.png" width="500">
<img src="https://github.com/issacto/Energy-Demand-Forecast/blob/master/Image/plottedgraph.png" width="500">

The model is quite accurate in that the loss is as 0.0007208.
