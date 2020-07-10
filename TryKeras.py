import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GaussianNoise
from keras.layers import Dense
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from keras.layers import LSTM


#Variables
dataset=np.loadtxt("windowWith10Inputs.csv", delimiter=",")
#split the data into input and output
x=dataset[:,0:10]
y=dataset[:,10]
y=np.reshape(y, (-1,1))


#split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = None)
n_features = 1

#Normalization and reshape
scaler_x_train = MinMaxScaler()
scaler_y_train = MinMaxScaler()
scaler_x_test = MinMaxScaler()
scaler_y_test = MinMaxScaler()

scaler_x_train.fit(X_train)
scaler_y_train.fit(y_train)
scaler_x_test.fit(X_test)
scaler_y_test.fit(y_test)

X_train_scaled = scaler_x_train.transform(X_train)
y_train_scaled = scaler_y_train.transform(y_train)
X_test_scaled = scaler_x_test.transform(X_test)
y_test_scaled = scaler_y_test.transform(y_test)

X_train_scaled = X_train_scaled.reshape(X_train.shape[0],X_train.shape[1],1)
X_test_scaled = X_test_scaled.reshape(X_test.shape[0],X_test.shape[1],1)


#Build model
model = Sequential()
model.add(LSTM(10,  activation='tanh', input_shape=(10, 1),return_sequences=True))
#model.add(Dense(10, activation='relu'))
model.add(LSTM(8, activation='tanh'))
model.add(Dropout(0.01))
model.add(Dense(1, activation='linear'))
model.summary()

opt = optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=opt)
es = EarlyStopping(monitor='val_loss',patience=3)

#run model
history = model.fit(X_train_scaled, y_train_scaled, epochs=20,validation_split = 0.2, batch_size=32,callbacks=[es])
print(history.history)
#plot graphs regarding the results
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()

#Evaluate the model on the test data
print("Evaluate on test data")
result = model.evaluate(X_test_scaled, y_test_scaled, batch_size=32)
print(str(model.metrics_names), result)


#try to plot a graph comparing prediction vs actual of the test data

#store the actual ouput of the testdata
test_prediction_array =[]
for x in range(len(X_test_scaled)):
    Xnew = np.array([X_test_scaled[x]])
    ynew= model.predict(Xnew)
    #invert normalize
    # Xnew = scaler_x_test.inverse_transform(Xnew[0])
    ynew = scaler_y_test.inverse_transform(ynew) 
    test_prediction_array.append(ynew[0])
    # print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
test_actual_data = scaler_y_test.inverse_transform(y_test_scaled)
z =[]
for i in range(len(test_prediction_array)):
    z.append(abs(test_prediction_array[i]-test_actual_data[i])/test_actual_data[i])
plt.plot(z)
plt.show()


#Plot graph out
print(test_prediction_array)
test_actual_data = scaler_y_test.inverse_transform(y_test) 
plt.plot(test_prediction_array[0:200])
plt.plot(test_actual_data[0:200])
plt.title('prediction vs actual')
plt.ylabel('value')
plt.xlabel('epoch')
plt.legend(['prediction', 'actual'], loc='upper left')
plt.show()

