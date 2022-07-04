import tensorflow as tf
import numpy as np
from typing import List
from Node import Node
from abc import ABC, abstractmethod

# USING PADDED TREES
class TreeRNN(ABC):
    def __init__(self, trees: List[Node], labels, layers, activationFunction: str, learningRate: float, 
    epochs: int):
        self.trees = trees
        self.labels = labels
        self.layers = layers
        self.layerCount = len(self.layers)
        self.treeCount = len(self.trees)
        self.activationFunction = self.getActivationFunction(activationFunction)
        self.learningRate = learningRate
        self.epochs = epochs
        self.weights, self.bias, self.weightDeltas, self.biasDeltas = {}, {}, {}, {}

    def getActivationFunction(self, activationFunction: str):
        if activationFunction == 'softmax':
            return tf.nn.softmax
        elif activationFunction == 'relu':
            return tf.nn.relu
        elif activationFunction == 'tanh':
            return tf.tanh
        elif activationFunction == 'logsigmoid':
            def logSigmoid(x):
                x = 1.0/(1.0 + tf.math.exp(-x)) 
                return x
            return logSigmoid
        else:
            return None

    def lossFunction(self, outputs, yValues):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yValues, outputs))

    def makePrediction(self, x_test):
        predictions = []
        for tree in x_test:
            output = self.RNNLayer(tree)
            prediction = tf.argmax(tf.nn.softmax(output), axis=1)
            predictions.append(prediction)
        return tf.convert_to_tensor(predictions)

    @abstractmethod
    def initialiseWeights(self):
        raise NotImplementedError()

    @abstractmethod
    def RNNLayer(self, tree, treeCount = None):
        raise NotImplementedError()

    @abstractmethod
    def backPropagate(self, tree, yValues, treeCount = None):
        raise NotImplementedError()

    @abstractmethod
    def updateWeights(self, index: int = None):
        raise NotImplementedError()

    @abstractmethod
    def runModel(self, x_train, y_train, x_test, y_test):
        raise NotImplementedError()
