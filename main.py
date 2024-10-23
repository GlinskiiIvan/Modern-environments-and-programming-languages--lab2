import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

model = Sequential()
model.add(Dense(60, input_dim=60, init="normal", activation="relu"))
model.add(Dense(1, init="normal", activation="sigmoid"))

model.compile(optimizer='adam',loss='binary crossentropy', metrics=['accuracy'])

model.fit(X, encoded_Y, epochs=100, batch_size=10, validation_split=0.1)

