import tensorflow as tf
import networkx as nx
import numpy as np
from typing import List
from BinaryGraphInputLayer import BinaryGraphInputLayer as BGIL

class HiddenGraphLayer():
    def __init__(self, learningRate):
        self.learningRate = learningRate

    def getLayer(self, neurons: int, layerName: str, activationFunction: str, useBias: bool, dropoutRate=None):
        activationFunction = self.testActivationTypes(activationFunction)
        layerName = layerName.lower()
        if layerName == 'rnn':
            return tf.keras.layers.SimpleRNN(neurons, activation=activationFunction, use_bias=useBias)
        elif layerName == 'lstm':
            return tf.keras.layers.LSTM(neurons, activation=activationFunction, use_bias=useBias)
        elif layerName == 'ffn':
            return self.MLPLayer(2, [neurons, neurons], activationFunction)
        elif layerName == 'dropout':
            return self.addDropoutLayer(dropoutRate)
        elif layerName == 'output':
            return self.addDenseLayer(neurons, activationFunction, useBias)

    def MLPLayer(self, hiddenLayerCount: int, hiddenLayerUnits: List[int], activationFunction):
        if len(hiddenLayerUnits) != hiddenLayerCount:
            raise Exception("Something is wrong somewhere, check that again")

        self.layers = []
        for i in range(hiddenLayerCount): 
            self.layers.append(self.addDenseLayer(hiddenLayerUnits[i], useBias=True, activationFunction=activationFunction))
        return self.layers

    def addDropoutLayer(self, dropoutRate):
        if dropoutRate is None:
            dropoutRate = 0.3
        return tf.keras.layers.Dropout(dropoutRate)

    def addDenseLayer(self, neurons: int, activationFunction: str, useBias: bool):
        return tf.keras.layers.Dense(neurons, activationFunction, useBias)

    def testActivationTypes(self, activationFunction: str):
        activationFunction = activationFunction.lower()
        if activationFunction == 'softmax':
            return tf.nn.softmax
        elif activationFunction == 'relu':
            return tf.nn.relu
        elif activationFunction == 'tanh':
            return tf.tanh
        elif activationFunction == 'logsigmoid':
            def logSigmoid(x):
                weights = tf.Variable(tf.random.normal(shape=(len(x), 2)), dtype=np.float32)
                bias = tf.Variable(tf.random.normal(shape=(2, 1)), dtype=np.float32)
                x = tf.matmul(x, weights) + tf.transpose(bias)
                x = 1.0/(1.0 + tf.math.exp(-x)) 
                return x
            return logSigmoid

