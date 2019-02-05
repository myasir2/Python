# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
from keras.callbacks import TensorBoard

# Import Dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
train_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
scaler = MinMaxScaler()
scaled_train_set = scaler.fit_transform(train_set)

# Dataset with 60 timestamps and 1 output
x_train = []
y_train = []

for i in range(60, 1258):
    x_train.append(scaled_train_set[i-60:i, 0])
    y_train.append(scaled_train_set[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# RNN
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
#model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
#model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
#model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = False))
#model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer = "adam", loss = "mean_squared_error") 

tensorboard = TensorBoard(log_dir='./tensorboard', histogram_freq=0, write_graph=True, write_images=False)

history = model.fit(x_train, 
                    y_train, 
                    epochs = 100, 
                    batch_size = 32, 
                    callbacks = [tensorboard],
                    validation_split = 0.33)

model.save('model.h5')
#model = load_model('model.h5')

# Testing the RNN
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Prepare the test set
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)
x_test = []

for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_stock_price = model.predict(x_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Visualization of predictions
plt.plot(real_stock_price, color = 'red', label = 'Real')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Visualization of model
plt.plot(history.history['loss'])
plt.title('model loss')
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# RMSE
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))








