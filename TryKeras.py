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
dataset=np.loadtxt("/Users/issac/Documents/GitHub/Energy-Demand-Forecast/windowWith10Inputs.csv", delimiter=",")
#split the data into input and output
x=dataset[:,0:10]
y=dataset[:,10]
y=np.reshape(y, (-1,1))
#Normalization
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(x))
xscale=scaler_x.transform(x)
print(scaler_y.fit(y))
yscale=scaler_y.transform(y)


#split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(xscale, yscale, test_size = 0.2, random_state = 0)
n_features = 1
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

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
history = model.fit(X_train, y_train, epochs=20,validation_data=(X_test, y_test), batch_size=32,callbacks=[es])

#plot graphs regarding the results
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='train')
plt.legend()
plt.show()


'''
#try to plot a graph comparing prediction vs actual of the test data

#store the actual ouput of the testdata
test_prediction_array =[]
for x in range(len(X_test)):
    Xnew = np.array([X_test[x]])
    ynew= model.predict(Xnew)
    #invert normalize
    ynew = scaler_y.inverse_transform(ynew) 
    Xnew = scaler_x.inverse_transform(Xnew)
    test_prediction_array.append(ynew[0])
    print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
test_actual_data = scaler_y.inverse_transform(y_test)
z =[]
for i in range(len(test_prediction_array)):
    z.append(abs(test_prediction_array[i]-test_actual_data[i])/test_actual_data[i])
plt.plot(z)
plt.show()
'''

#Plot graph out
'''
test_actual_data = scaler_y.inverse_transform(y_test) 
plt.plot(test_prediction_array[0:200])
plt.plot(test_actual_data[0:200])
plt.title('prediction vs actual')
plt.ylabel('value')
plt.xlabel('epoch')
plt.legend(['prediction', 'actual'], loc='upper left')
plt.show()
'''