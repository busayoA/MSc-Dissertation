import readFiles as rf
from keras.models import Sequential
from keras.layers import Input, Activation, Dense, Dropout, LSTM, add, Bidirectional


x_train, y_train, x_test, y_test = rf.getVectorizedCodeData()

filename = "otherLabelRNNModel.hdf5"

model = Sequential()

model.add(Input(shape=(x_train.shape[1], 1)))
model.add(Bidirectional(LSTM(256, input_shape=(x_train.shape[0], 1), return_sequences=True)))
model.add(LSTM(256))
model.add(Dropout(0.3))
model.add(Dense(3, activation='relu'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

model.save(filename)
