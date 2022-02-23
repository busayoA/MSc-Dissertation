import NeuralNetworks.RecurrentNeuralNetwork as Network
import readFiles as rf

x_train, y_train, x_test, y_test = rf.createTrainTestData()
rnn = Network.RecurrentNeuralNetwork(20, [60, 128, 128, 5]) 
print(rnn.activationFunction())
