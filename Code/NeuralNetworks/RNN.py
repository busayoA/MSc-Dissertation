#!/usr/local/bin/python3

import tensorflow as tf
import numpy as np

class RNN:
    def __init__(self, layers):
        self.layers = layers
        self.layerCount = len(layers)
        self.featureCount = layers[0]
        self.classCount = layers[-1]

        self.weights = {}
        self.bias = {}
        
    def setupModel(self):
        for i in range(1, self.layerCount):
            self.weights[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], self.layers[i-1])))
            self.bias[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], 1)))
        
    def getModelInfo(self):
        pass
    
    def forwardPass(self):
        pass

    def backwardPass(self):
        pass

    def computeLoss(self):
        pass

    def updateParameters(self):
        pass

    def trainModel(self):
        pass

    def makePrediction(self):
        pass

    