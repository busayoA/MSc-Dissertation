import NeuralNetworks.RecurrentNeuralNetwork as Network
import preTraining as pt, readFiles as rf, tensorflow as tf
import numpy as np

x_train, y_train, x_test, y_test = rf.createTrainTestData()

x_train = pt.createEmbeddings(x_train)
x_test = pt.createEmbeddings(x_test)
x_train = np.reshape(x_train, (x_train.shape[0], len(x_train[0])))/255.
x_test =  np.reshape(x_test, (x_test.shape[0], len(x_test[0])))/255.
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# print(y_train)


rnn = Network.RecurrentNeuralNetwork(10, 0.3, 10, [len(x_train[0]), 10, 10, 5]) 

stepsPerEpoch = int(x_train.shape[0]/5)
print(x_train.shape[0])
print(" stepsPerEpoch: ", stepsPerEpoch)
measures = rnn.trainModel(x_train, y_train, x_test, y_test, stepsPerEpoch)
