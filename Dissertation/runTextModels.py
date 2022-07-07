from TextParser import TextParser
import numpy as np
import tensorflow as tf
from MLP import MLP
from RNN import RNN

tp = TextParser()
x_train, y_train, x_test, y_test = tp.getVectorizedTextData()

layers = [len(x_train[0]), 128, 128, 2]
epochs = 10
lr = 0.001
mlp1 = MLP(x_train, y_train, layers, "relu", lr, epochs)
metrics1 = mlp1.runFFModel(x_train, y_train, x_test, y_test)

mlp2 = MLP(x_train, y_train, layers, "tanh", lr, epochs)
metrics2 = mlp2.runFFModel(x_train, y_train, x_test, y_test)

mlp3 = MLP(x_train, y_train, layers, "softmax", lr, epochs)
metrics3 = mlp3.runFFModel(x_train, y_train, x_test, y_test)

mlp4 = MLP(x_train, y_train, layers, "logsigmoid", lr, epochs)
metrics4 = mlp4.runFFModel(x_train, y_train, x_test, y_test)


print("USING THE MULTI-LAYER PERCEPTRON")
print("USING RELU")
print("Average loss:", np.average(metrics1['trainingLoss']), "Average training accuracy:", 
np.average(metrics1['trainingAccuracy']), "Average validation accuracy:", 
np.average(metrics1['validationAccuracy']), "\n") 

print("USING TANH")
print("Average loss:", np.average(metrics2['trainingLoss']), "Average training accuracy:", 
np.average(metrics2['trainingAccuracy']), "Average validation accuracy:", 
np.average(metrics2['validationAccuracy']), "\n") 

print("USING SOFTMAX")
print("Average loss:", np.average(metrics3['trainingLoss']), "Average training accuracy:", 
np.average(metrics3['trainingAccuracy']), "Average validation accuracy:", 
np.average(metrics3['validationAccuracy']), "\n") 

print("USING SIGMOID")
print("Average loss:", np.average(metrics4['trainingLoss']), "Average training accuracy:", 
np.average(metrics4['trainingAccuracy']), "Average validation accuracy:", 
np.average(metrics4['validationAccuracy']), "\n") 

dimensions = tf.shape(x_train)
x_train = tf.reshape(x_train, (dimensions[0], dimensions[1]))
inputLayer = tf.keras.layers.InputLayer(input_shape=(x_train.shape[1], 1))
inputShape=(x_train.shape[0], )

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

model1.fit(x_train, y_train, epochs=10, batch_size=5, validation_data=(x_test, y_test))

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

model2.fit(x_train, y_train, epochs=10, batch_size=5, validation_data=(x_test, y_test))

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

model3.fit(x_train, y_train, epochs=10, batch_size=5, validation_data=(x_test, y_test))

model3.save(filename3)
