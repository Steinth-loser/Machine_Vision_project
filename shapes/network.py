import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

dataset = pd.read_csv('data/shapes_dataset.csv')
dataset

x_train = dataset.drop('label', axis=1)
x_train_array = x_train.values
x_train_array = x_train_array.reshape(-1, 28, 28, 1)
y_train = dataset['label']



model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(784, activation='relu'))
model.add(Dense(28,activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 classes

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(x_train_array, y_train, epochs=20, batch_size=32)

model.save('trained_model.h5')
