import tensorflow as tf
import numpy as np
from BinaryGraphInputLayer import BinaryGraphInputLayer as BGIL
from HiddenGraphLayer import HiddenGraphLayer as HGL


merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort"
quick = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Quick Sort"

inputLayer = BGIL()
xTrain, yTrain, xTest, yTest = inputLayer.splitTrainTest(merge, quick)
x_train, x_train_matrix, y_train, x_test, x_test_matrix, y_test = inputLayer.getDatasets(xTrain, yTrain, xTest, yTest)
# x_train_0 = inputLayer.getAdjacencyLists(x_train[0], x_train_matrix[0])

print("Collecting node embeddings and adjacency lists")
x_train_adj = [0] * len(x_train)
for i in range(len(x_train)):
    x_train[i], x_train_adj[i] = inputLayer.getAdjacencyLists(x_train[i], x_train_matrix[i])

ffnModel = HGL(0.03)
layerName = "ffn"
activationFunction = "logSigmoid"

def graphFFNLayer(layerName, activationFunction, x_train_tuple):
    # ffnLayers = ffnModel.getLayer(128, layerName, activationFunction, True)
    
    # hidden1 = ffnLayers[0]
    # hidden2 = ffnLayers[1]
    print(x_train_tuple[0][0])


graphFFNLayer(layerName, activationFunction, x_train[0])
# out = hiddenLayer.forwardPropagate(x_train[0], x_train_matrix[0])
# loss = hiddenLayer.lossFunction(out, y_train[0])
# print(loss.numpy())