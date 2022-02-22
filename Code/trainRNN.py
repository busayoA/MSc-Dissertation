import readFiles as rf
from NeuralNetworks import RNN

x_train, y_train, x_test, y_test = rf.createTrainTestData()
rNeuralNet = RNN.RNN([60, 128, 128, 5])
rNeuralNet.setupModel()