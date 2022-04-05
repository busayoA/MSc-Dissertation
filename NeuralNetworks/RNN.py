from math import exp
import numpy as np
import random
from statistics import mean

class RNN:
    def __init__(self, layers, epochs, lr):
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.layerCount = len(layers)
        self.hiddenLayerCount = len(layers)-2
        self.featureCount = layers[0]
        self.classCount = layers[-1]
        self.parameterCount = 0
        self.weights, self.weightErrors, self.bias, self.biasErrors = [], [], [], []

        # Set up the model based on the number of layers (minus the input layer):
        for i in range(1, self.layerCount):
            self.weights.append([[random.random() for k in range(self.featureCount+1)] for j in range(self.hiddenLayerCount)])
            self.bias.append([[random.random() for k in range(self.hiddenLayerCount+1)] for j in range(self.classCount)])

        print(self.featureCount, "features,", self.classCount, "classes", "and", self.hiddenLayerCount, "hidden layers", "\n")

        for i in range(1, self.layerCount-1):
            print('Hidden layer {}:,'.format(i), '{} neurons'.format(self.layers[i]))

    def forwardPass(self, featureValues):
        values, finalValues = [], featureValues
        for x in range(1, self.layerCount):
            currentLayer = self.layers[x]
            for j in range(currentLayer):
                finalValue, val = 0.0, 0.0
                for j in range(len(self.weights[x-1])):
                    try:
                        val += sum(self.weights[x-1][j])
                    except:
                        val += self.weights[x-1][j]
                    activationValue = 1.0 / (1.0 + exp(-finalValues[j]))
                    transferValues = random.random() * activationValue
                for i in range(len(finalValues)):
                    finalValue = transferValues * finalValues[i]
                if x == self.layerCount-1:
                    values.append(finalValue)
        return values   

    def backwardPass(self, expectedOutput, actualValues):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            errorList = []
            biasErrorList = []
            if i is not len(self.layers)-1:
                for j in range(layer):
                    error = 0.0
                    for k in range(self.layers[i+1]):
                        transferValue =  mean(expectedOutput) * (1.0 - mean(expectedOutput))
                        error += transferValue
                        errorList.append(error)
                self.weightErrors.append(mean(errorList))
            else:
                for j in range(layer):
                    biasErrorList.append(actualValues[j] - expectedOutput[j])
                self.biasErrors.append(mean(biasErrorList))

    def updateParameters(self):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] = (self.lr * self.weightErrors[len(self.weightErrors)-1])
            for j in range(len(self.bias[i])):
                self.bias[i][j] = (self.lr *  self.biasErrors[len(self.biasErrors)-1])

    def trainModel(self, x_train, y_train, x_test, y_test):
        for epoch in range(self.epochs):
            error = 0
            index = 0
            for features in x_train:
                output = self.forwardPass(features)
                expected = [0] * self.classCount
                expected[y_train[index]] = 1
                error += sum([(expected[i]-output[i])**2 for i in range(len(expected))])
                self.backwardPass(expected, output)
                self.updateParameters()
                index += 1

            validationAccuracy = 0
            for i in range(len(x_test)):
                prediction = self.predict(x_test[i])
                if (prediction[0][0] == y_test[i]):
                    validationAccuracy += 1

            validationAccuracy = validationAccuracy/len(x_test)

            error = error/len(x_train)
            print("Epoch:", epoch, "Error:", error, "Accuracy:", validationAccuracy)

    def predict(self, featureValues):
        outputs = self.forwardPass(featureValues)
        return np.where(outputs == np.amax(outputs))
        