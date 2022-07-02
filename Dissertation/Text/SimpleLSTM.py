import readFiles as rf
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, LSTM, Bidirectional


# BINARY MODEL
x_train, y_train, x_test, y_test = rf.getVectorizedCodeData(False)

filename1 = "binaryRNNModel.hdf5"

model1 = Sequential()

model1.add(Input(shape=(x_train.shape[1], 1)))
model1.add(Bidirectional(LSTM(256, input_shape=(x_train.shape[0], 1), return_sequences=True)))
model1.add(LSTM(256))
model1.add(Dropout(0.3))
model1.add(Dense(2, activation='sigmoid'))

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model1.summary())

model1.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

model1.save(filename1)


# BINARY MODEL
x_train, y_train, x_test, y_test = rf.getVectorizedCodeData(True)

filename2 = "otherLabelRNNModel.hdf5"

model2 = Sequential()

model2.add(Input(shape=(x_train.shape[1], 1)))
model2.add(Bidirectional(LSTM(256, input_shape=(x_train.shape[0], 1), return_sequences=True)))
model2.add(LSTM(256))
model2.add(Dropout(0.3))
model2.add(Dense(3, activation='relu'))

model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model2.summary())

model2.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

model2.save(filename2)
