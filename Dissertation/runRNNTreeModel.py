import tensorflow as tf
import TreeSegmentation as seg
from RNN import RNN

x_train_usum =  seg.x_train_usum
x_train_umean = seg.x_train_umean
x_train_umax = seg.x_train_umax
x_train_umin = seg.x_train_umin
x_train_uprod = seg.x_train_uprod

x_test_usum = seg.x_test_usum
x_test_umean = seg.x_test_umean
x_test_umax = seg.x_test_umax
x_test_umin = seg.x_test_umin
x_test_uprod = seg.x_test_uprod

x_train_sum = seg.x_train_sum
x_train_mean = seg.x_train_mean
x_train_max = seg.x_train_max
x_train_min = seg.x_train_min
x_train_prod = seg.x_train_prod

x_test_sum = seg.x_test_sum
x_test_mean = seg.x_test_mean
x_test_max = seg.x_test_max
x_test_min = seg.x_test_min
x_test_prod = seg.x_test_prod

y_train = seg.y_train
y_test = seg.y_test


inputLayer = tf.keras.layers.InputLayer(input_shape=(x_train_usum.shape[1], 1))
inputShape=(x_train_usum.shape[0], )

gruModel = RNN("gru", activationFunction="relu", neurons=256)
gruLayer = gruModel.chooseModel(inputShape=inputShape, returnSequences=True)

lstmModel = RNN("lstm", activationFunction="relu", neurons=256)
lstmLayer = lstmModel.chooseModel(inputShape=inputShape, returnSequences=True)

rnnModel = RNN("rnn", activationFunction="relu", neurons=256)
rnnLayer = rnnModel.chooseModel()

dropout = RNN("dropout", activationFunction="relu", dropoutRate=0.3)
dropout = dropout.chooseModel()

denseModel = RNN("dense", activationFunction="relu", neurons=2)
denseLayer = denseModel.chooseModel()


print("USING THE RECURRENT NEURAL NETWORK")
print("USING LSTM LAYERS")
filename1 = "textLSTMModel.hdf5"
model1 = tf.keras.models.Sequential()

model1.add(inputLayer)
model1.add(tf.keras.layers.Bidirectional(lstmLayer))
model1.add(tf.keras.layers.LSTM(256))
model1.add(dropout)
model1.add(tf.keras.layers.Dense(2, activation='sigmoid'))

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model1.summary()

model1.fit(x_train_usum, y_train, epochs=10, batch_size=5, validation_data=(x_test, y_test))

model1.save(filename1)


print("USING GRU LAYERS")
filename2 = "textGRUModel.hdf5"
model2 = tf.keras.models.Sequential()

model2.add(inputLayer)
model2.add(tf.keras.layers.Bidirectional(gruLayer))
model2.add(tf.keras.layers.GRU(256))
model2.add(dropout)
model2.add(tf.keras.layers.Dense(2, activation='sigmoid'))

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model2.summary()

model2.fit(x_train_usum, y_train, epochs=10, batch_size=5, validation_data=(x_train_usum, y_test))

model2.save(filename2)


print("USING SIMPLE RNN LAYERS")
filename3 = "textRNNModel.hdf5"
model3 = tf.keras.models.Sequential()

model3.add(inputLayer)
model3.add(tf.keras.layers.Bidirectional(rnnLayer))
model3.add(dropout)
model3.add(tf.keras.layers.Dense(2, activation='sigmoid'))

model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model3.summary()

model3.fit(x_train_usum, y_train, epochs=10, batch_size=5, validation_data=(x_train_usum, y_test))

model3.save(filename3)



