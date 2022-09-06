# Minicourse July 26th - July 30th
# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# Part 1 - Data Preprocessing
# Importing the Data
path = 'Churn_Modelling.csv' 
dataset = pd.read_csv(path,engine='python')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding the Categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN
from tensorflow.keras import layers
from tensorflow.keras import activations
# Initializing the ANN
ann = tf.keras.models.Sequential()

# Add the input layer and the first hidden layer
ann.add(tf.keras.Input(shape=(12,)))
ann.add(tf.keras.layers.Dense(32,activation='relu'))

# Add the second hidden layer
ann.add(tf.keras.layers.Dense(16, activation='relu'))

# Add the output layer
ann.add(tf.keras.layers.Dense(1,activation='tanh'))

# Part 3 - Training the ANN
# Compiling the ANN
ann.compile(optimizer='adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# Part 4 - Making the predictions and evaluating the model
# Predicting the result of a single observation
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)