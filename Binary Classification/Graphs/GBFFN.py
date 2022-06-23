import tensorflow as tf
import numpy as np
from BinaryGraphInputLayer import BinaryGraphInputLayer as BGIL
from HiddenGraphLayer import HiddenGraphLayer as HGL


merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort"
quick = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Quick Sort"

inputLayer = BGIL()
xTrain, yTrain, xTest, yTest = inputLayer.splitTrainTest(merge, quick)
x_train, x_train_matrix, y_train, x_test, x_test_matrix, y_test = inputLayer.getDatasets(xTrain, yTrain, xTest, yTest)
xTrainTest, xTrainTestAdj = inputLayer.getAdjacencyLists(x_train[0], x_train_matrix[0])

hiddenFFNLayer1 = HGL(0.03)
layerName = "ffn"
activationFunction = "logSigmoid"

def graphFFNLayer(layerName, activationFunction, xTrain, xTrainAdjacencies):
    ffnLayer = hiddenFFNLayer1.getLayer(128, layerName, activationFunction, True)
    # print(ffnLayer)




graphFFNLayer(layerName, activationFunction, xTrainTest, xTrainTestAdj)
# out = hiddenLayer.forwardPropagate(x_train[0], x_train_matrix[0])
# loss = hiddenLayer.lossFunction(out, y_train[0])
# print(loss.numpy())