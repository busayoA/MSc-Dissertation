from platform import architecture
from cv2 import exp
import tensorflow as tf
import numpy as np

class RecurrentNeuralNetwork:
    def __init__(self, epochs, learningRate, batchSize, values):
        self.epochs = epochs
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.values = values
        self.layerCount = len(values)
        self.hiddenLayerCount = len(values)-2
        self.featureCount = values[0]
        self.classCount = values[-1]
        self.parameterCount = 0
        self.weights, self.weightErrors, self.bias, self.biasErrors = {}, {}, {}, {}

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


    def feedForward(self, weights):
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        for i in range(1, self.layerCount):
            multWeights = tf.matmul(weights, tf.transpose(self.weights[i])) + tf.transpose(self.bias[i])
            if i is not self.layerCount-1:
                weights = tf.nn.relu(multWeights)
            else:
                weights = multWeights
        return weights
    

    def backPropagateWeights(self, learningRate):
        #carry out back propagation on the weights and biases and update the weights
        for i in range(1, self.layerCount):
            self.weights[i] = self.weights[i].assign_sub(learningRate * self.weightErrors[i])
            self.bias[i] = self.bias[i].assign_sub(learningRate * self.biasErrors[i])


    def computeLoss(self, xValues, yValues):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yValues, xValues))


    def trainModel(self, x_train, y_train, x_test, y_test, epochs, learningRate):
        self.stepsPerEpoch = int(x_train.shape[0]/self.batchSize)


    def makePrediction(self):
        pass

    
