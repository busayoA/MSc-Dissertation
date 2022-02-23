from platform import architecture
from cv2 import exp
import tensorflow as tf
import numpy as np

class RecurrentNeuralNetwork:
    def __init__(self, epochs, learningRate, values):
        self.epochs = epochs
        self.learningRate = learningRate
        self.values = values
        self.layerCount = len(values)
        self.hiddenLayerCount = len(values)-2
        self.featureCount = values[0]
        self.classCount = values[-1]
        self.activationValues = []
        self.parameterCount = 0
        self.weights = {}
        self.bias = {}

        # Set up the model based on the number of layers (minus the input layer):
        for i in range(1, self.layerCount):
            self.weights[i] = tf.Variable(tf.random.normal(shape=(self.values[i], self.values[i-1])))
            self.bias[i] = tf.Variable(tf.random.normal(shape=(self.values[i], 1)))
        
        
        for i in range(1, self.layerCount):
            self.parameterCount += self.weights[i].shape[0] * self.weights[i].shape[1]
            self.parameterCount += self.bias[i].shape[0]
        print(self.featureCount, "features,", self.classCount, "classes,", self.parameterCount, "parameters, and",
            self.layerCount-1, "hidden layers", "\n")
        for i in range(1, self.layerCount-1):
            print('Hidden layer {}:,'.format(i), '{} neurons'.format(self.values[i]))

        # print(self.bias)


    def activationFunction(self):
        for i in range(1, self.layerCount):
            self.activationValues.append(self.bias[i] + (self.weights[i].shape[0] * self.values[i]))
        return self.activationValues

    def transferActivation(self, activationValue):
        return (1.0/ (1.0 * exp(activationValue)))

    def forwardPass(self, weights):
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        for i in range(1, self.layerCount):
            multWeights = tf.matmul(weights, tf.transpose(self.weights[i])) + tf.transpose(self.bias[i])
            if i is not self.layerCount-1:
                weights = tf.nn.relu(multWeights)
            else:
                weights = multWeights
        return weights

    def forwardPassOne(self, values):
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

    
