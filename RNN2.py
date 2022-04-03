from distutils.log import error
import math
import random
import numpy as np
import readFiles as RF, preTraining as PT

class RNN2():
    def __init__(self, inputCount, hiddenCount, outputCount):
        self.inputCount = inputCount
        self.hiddenCount = hiddenCount
        self.outputCount = outputCount
        self.layers = list()
        self.hiddenLayers = [{'weights':[random.random() for i in range(inputCount + 1)]} for i in range(hiddenCount)]
        self.layers.append(self.hiddenLayers)
        self.outputLayer = [{'weights':[random.random() for i in range(hiddenCount + 1)]} for i in range(outputCount)]
        self.layers.append(self.outputLayer)

        print(self.layers)

    def activateWeights(self, weights, values):
        activationValue = weights[-1]
        for i in range(len(weights)-1):
            if len(values)-1 < i:
                activationValue += weights[i] 
            else:
                activationValue += weights[i] * values[i]

        return activationValue

    def activationFunction(self, activationValue):
        return 1.0 / (1.0 + math.exp(-activationValue))

    def transferFunction(self, outputValue):
        return outputValue * (1.0 - outputValue)

    def forwardPass(self, row):
        values = row
        for layer in self.layers:
            propagatedValues = []
            for unit in layer:
                activationValue = self.activateWeights(unit['weights'], values)
                unit['output'] = self.transferFunction(activationValue)
                propagatedValues.append(unit['output'])
                values = propagatedValues
        return values

    def backwardPass(self, expectedOutput):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            errorList = []
            if i is not len(self.layers)-1:
                for j in range(len(layer)):
                    thisError = 0.0
                    for unit in self.layers[i+1]:
                        thisError += (unit['weights'][j] * unit['delta'])
                    errorList.append(thisError)
            else:
                for j in range(len(layer)):
                    unit = layer[j]
                    errorList.append(unit['output'] - expectedOutput[j])
            for j in range(len(layer)):
                unit = layer[j]
                unit['delta'] = errorList[j] * self.transferFunction(unit['output'])

    def updateParameters(self, row, learningRate):
        for i in range(len(self.layers)):
            values = row[:-1]
            if i is not 0:
                values = [unit['output'] for unit in self.layers[i - 1]]
            for unit in self.layers[i]:
                for j in range(len(values)):
                    unit['weights'][j] -= learningRate * unit['delta'] * values[j]
                unit['weights'][-1] -= learningRate * unit['delta']

    def train(self, x_train, learningRate, epochs, outputCount):
        for epoch in range(epochs):
            error = 0
            for row in x_train:
                output = self.forwardPass(row)
                expected = [0 for i in range(outputCount)]
                expected[row[-1]] = 1
                error += sum([(expected[i]-output[i])**2 for i in range(len(expected))])
                self.backwardPass(expected)
                self.updateParameters(row, learningRate)


a, b, c, d = (RF.createTrainTestData())
xTrainUntagged = PT.createEmbeddings(a)
xTestUntagged = PT.createEmbeddings(c)

x_train = [0] * len(xTrainUntagged)
x_test = [0] * len(xTestUntagged)


for i in range(len(xTrainUntagged)):
    x_train[i] = np.append(xTrainUntagged[i], b[i])
    print(x_train[i])

x_train = np.array(x_train)
# print(x_train)

for i in range(len(xTestUntagged)):
    x_test[i] = np.append(xTestUntagged[i], d[i])
    print(x_test[i])

x_test = np.array(x_test)
# print(x_test)


# layers = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}], [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
# expected = [0, 1]
# rnn.backwardPass(layers, expected)
# for layer in rnn.layers:
#     print(layer)

inputCount = len(x_train[0]-1)
outputCount = len(set([row[-1] for row in x_train]))
rnn = RNN2(inputCount, 2, outputCount)
rnn.train(x_train, 0.3, 20, outputCount)
