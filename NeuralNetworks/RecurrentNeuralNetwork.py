from platform import architecture
from turtle import forward
from cv2 import exp
import tensorflow as tf
import numpy as np
from torch import reshape

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
        # weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        for i in range(1, self.layerCount):
            multWeights = tf.matmul(weights, tf.transpose(self.weights[i])) + tf.transpose(self.bias[i])
            if i is not self.layerCount-1:
                weights = tf.nn.relu(multWeights)
            else:
                weights = multWeights
        return weights
    

    def backPropagateWeights(self):
        #carry out back propagation on the weights and biases and update the weights
        for i in range(1, self.layerCount):
            self.weights[i] = self.transferDerivatives(self.weights[i])
            self.bias[i] = self.transferDerivatives(self.bias[i])


    def computeLoss(self, xValues, yValues):
        return self.transferDerivatives(xValues)

    def transferDerivatives(self, weights):
        return weights * (1.0 - weights)

    def makePrediction(self, xValues):
        prediction = self.feedForward(xValues)
        return tf.argmax(tf.nn.softmax(prediction), axis=1)

    def trainModelByBatch(self, xValues, yValues):
        xValues = tf.convert_to_tensor(xValues, dtype=tf.float32)
        yValues = tf.convert_to_tensor(yValues, dtype=tf.float32)

        forwardXValues = self.feedForward(xValues)
        loss = self.computeLoss(forwardXValues, yValues)

        self.backPropagateWeights()
        
        return loss

    def trainModel(self,  x_train, y_train, x_test, y_test, stepsPerEpoch):
        self.stepsPerEpoch = stepsPerEpoch
        measures = {'Validation Loss': [], 'Training Loss': [], 'Validation Accuracy': []}
        
        for i in range(0, self.epochs):
            trainingLoss = 0.
            print('Epoch {}'.format(i), end='.')
            for j in range(0, stepsPerEpoch):
                x_batch = x_train[j*self.batchSize:(j+1)*self.batchSize]
                y_batch = y_train[j*self.batchSize:(j+1)*self.batchSize]
                batch_loss = self.trainModelByBatch(x_batch, y_batch)
                trainingLoss += batch_loss
                
                if i%int(stepsPerEpoch/10) == 0:
                    print(end='.')
                    
            measures['Training Loss'].append(trainingLoss/stepsPerEpoch)
            
            
            testValidation = self.feedForward(x_test)
            measures['Validation Loss'].append(self.compute_loss(testValidation, y_test).numpy())
            
            predictionValidation = self.makePrediction(x_test)
            measures['Validation Accuracy'].append(np.mean(np.argmax(y_test, axis=1) == predictionValidation.numpy()))
            print('Validation Accuracy = ', measures['Validation Accuracy'][-1])
        return measures

    
