import numpy as np
import tensorflow as tf
import readTextFiles as rtf
import readCodeFiles as rcf
import matplotlib.pyplot as plt
from keras.utils import np_utils as nUtils
from keras.models import Sequential
from keras.layers import Input, Activation, Dense, Dropout, LSTM, Embedding, add, Bidirectional



# # ----------------------------------------------------------------------------------------------------------------------------------------------------------
x_train, y_train, x_test, y_test = rtf.getVectorizedData()
filename = "modelWithText.hdf5"

model = Sequential()
model.add(Input(shape=(x_train.shape[1], 1)))
model.add(Bidirectional(LSTM(256, input_shape=(x_train.shape[0], 1), return_sequences=True)))
model.add(LSTM(256))
model.add(Dropout(0.3))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
# model.save(filename)
# # ----------------------------------------------------------------------------------------------------------------------------------------------------------

# # ----------------------------------------------------------------------------------------------------------------------------------------------------------
x_train, y_train, x_test, y_test = rcf.getVectorizedData()
filename1 = "modelWithCode.hdf5"

model1 = Sequential()
model1.add(Input(shape=(x_train.shape[1], 1)))
model1.add(Bidirectional(LSTM(256, input_shape=(x_train.shape[0], 1), return_sequences=True)))
model1.add(LSTM(256))
model1.add(Dropout(0.3))
model1.add(Dense(2, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model1.summary())
# model1.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
# model1.save(filename1)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

