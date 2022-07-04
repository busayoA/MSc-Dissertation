import tensorflow as tf
import numpy as np
from statistics import mean
from keras.models import Sequential
from keras.layers import Input, Bidirectional
from GraphInputLayer import GraphInputLayer as GIL
from HiddenGraphLayer import HiddenGraphLayer as HGL



input = GIL()
x, matrices, labels = input.readFiles(False)
x_train_all, x_train_matrix, y_train, x_test_all, x_test_matrix, y_test = input.splitTrainTest(x, matrices, labels)
x_train_nodes = []
x_train_graph = []
for i in range(len(x_train_all)):
    x_train_nodes.append(x_train_all[i][0])
    x_train_graph.append(x_train_all[i][1])

x_test_nodes = []
x_test_graph = []
for i in range(len(x_test_all)):
    x_test_nodes.append(x_test_all[i][0])
    x_test_graph.append(x_test_all[i][1])

x_train = input.prepareData(x_train_graph, x_train_matrix)
x_test = input.prepareData(x_test_graph, x_test_matrix)


def GBFFNModel(xTrain, yTrain, hiddenActivationFunction):
    for graph in xTrain:
        dimensions = tf.shape(graph, out_type=np.int32)

        
                
        # print(dimensions.numpy())

withSigmoid = GBFFNModel(x_train, y_train, "logSigmoid")
# withTanh = GBFFNModel(x_train, y_train, "tanh")
# withSoftmax = GBFFNModel(x_train, y_train, "softmax")
# withRelu = GBFFNModel(x_train, y_train, "relu")

# print("With Sigmoid:", withSigmoid, "\nWith Tanh:", withTanh, "\nWith Softmax:", withSoftmax, "\nWith Relu:", withRelu)

