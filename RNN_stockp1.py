# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from pathlib import Path

# Importing the training set
path = 'Google_Stock_Price_Train.csv'
dataset_train = pd.read_csv(path)
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN
from tensorflow import keras
from tensorflow.keras import layers

# Initialize the RNN
rnn = tf.keras.Sequential()

# Adding the first LSTM layer and some Dropout regularisation
rnn.add(tf.keras.layers.LSTM(units = 50, return_sequences= True, input_shape = (X_train.shape[1], 1)))
rnn.add(tf.keras.layers.Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
rnn.add(layers.LSTM(units = 128, return_sequences = True))
rnn.add(layers.Dropout(0.5))

# Adding a third LSTM layer and some Dropout regularisation
rnn.add(layers.LSTM(units = 64, return_sequences = True))
#rnn.add(layers.LSTM(50))
rnn.add(layers.Dropout(0.3))


# Adding a fourth LSTM layer and some Dropout regularisation
rnn.add(layers.LSTM(units = 16))
rnn.add(layers.Dropout(0.1))

# Adding the output layer
rnn.add(layers.Dense(units = 1,activation='tanh'))

# Compiling the RNN
rnn.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
rnn.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
path = 'Google_Stock_Price_Test.csv'
dataset_test = pd.read_csv(path)
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = rnn.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()