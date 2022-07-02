import tensorflow as tf
import numpy as np
from statistics import mean
from keras.models import Sequential
from keras.layers import Input, Bidirectional
from GraphInputLayer import GraphInputLayer as GIL
from HiddenGraphLayer import HiddenGraphLayer as HGL



input = GIL()
x_train_nodes, x_train_matrix, y_train = input.readFiles(False)
x_train = input.prepareData(x_train_nodes, x_train_matrix)


def GBFFNModel(xTrain, yTrain, hiddenActivationFunction):
    for graph in xTrain:
        dimensions = tf.shape(graph, out_type=np.int32)

        
                
        # print(dimensions.numpy())

withSigmoid = GBFFNModel(x_train, y_train, "logSigmoid")
# withTanh = GBFFNModel(x_train, y_train, "tanh")
# withSoftmax = GBFFNModel(x_train, y_train, "softmax")
# withRelu = GBFFNModel(x_train, y_train, "relu")

# print("With Sigmoid:", withSigmoid, "\nWith Tanh:", withTanh, "\nWith Softmax:", withSoftmax, "\nWith Relu:", withRelu)

