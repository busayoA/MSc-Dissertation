import tensorflow as tf
import numpy as np
from typing import List
import TreeEmbeddingLayer as embeddingLayer
from TreeEmbeddingLayer import TreeEmbeddingLayer as tel
from Node import Node

# USING PADDED TREES
class TreeRNN1():
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

    def initialiseWeights(self):
        for i in range(1, self.layerCount):
            self.weights[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], self.layers[i-1])))
            self.bias[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], 1)))

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

    def testModelOnOneTree(self, treeEmbeddings, aggregationFunction):
        self.initialiseWeights()
        ffnCell = self.RNNLayer(treeEmbeddings, aggregationFunction)
        return ffnCell

    def RNNLayer(self, tree):
        outputs = tf.convert_to_tensor(tree, dtype=tf.float32)
        dimensions = tf.shape(outputs)
        outputs = tf.reshape(outputs, (1, dimensions[0]))
        for i in range(1, self.layerCount): 
            weights = self.weights[i]
            bias = self.bias[i]
            outputs = tf.matmul(outputs, tf.transpose(weights)) + tf.transpose(bias)
            predictions = self.activationFunction(outputs)
        return predictions

    def backPropagate(self, tree, yValues):
        with tf.GradientTape(persistent=True) as tape: 
            output = self.RNNLayer(tree)
            loss = self.lossFunction(output, yValues)
        
        for i in range(1, self.layerCount):
            self.weightDeltas[i] = tape.gradient(loss, self.weights[i])
            self.biasDeltas[i] = tape.gradient(loss, self.bias[i])
        del tape
        self.updateWeights()
        return loss.numpy()

    def updateWeights(self):
        for i in range(1, self.layerCount):
            self.weights[i].assign_sub(self.learningRate * self.weightDeltas[i])
            self.bias[i].assign_sub(self.learningRate * self.biasDeltas[i])

    def lossFunction(self, outputs, yValues):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yValues, outputs))

    def makePrediction(self, x_test):
        prediction = []
        predictions = []
        for tree in x_test:
            output = self.RNNLayer(tree)
            prediction = tf.argmax(tf.nn.softmax(output), axis=1)
            predictions.append(prediction)
        return tf.convert_to_tensor(predictions)

    def runModel(self, x_train, y_train, x_test, y_test):
        index = 0
        metrics = {'trainingLoss': [], 'trainingAccuracy': [], 'validationAccuracy': []}
        self.initialiseWeights()
        loss = []
        for i in range(self.epochs):
            predictions = []
            print('Epoch {}'.format(i), end='.')
            for tree in x_train:
                if index % 5 == 0:
                    print(end=".")
                if index >= len(y_train):
                    index = 0
                # FIRST FORWARD PASS
                loss.append(self.backPropagate(tree, y_train[index]))
                # SECOND FORWARD PASS/RECURRENT LOOP
                newOutputs = self.RNNLayer(tree)
                pred = tf.argmax(tf.nn.softmax(newOutputs), axis = 1)
                predictions.append(pred)
                index += 1
            predictions = tf.convert_to_tensor(predictions)
            # print(predictions)
            unseenPredictions = self.makePrediction(x_test)
            metrics['trainingLoss'].append(tf.reduce_mean(loss).numpy())
            metrics['trainingAccuracy'].append(np.mean(np.argmax(y_train, axis=1) == predictions.numpy()))
            metrics['validationAccuracy'].append(np.mean(np.argmax(y_test, axis=1) == unseenPredictions.numpy()))
            print('\tLoss:', metrics['trainingLoss'][-1], 'Training Accuracy:', metrics['trainingAccuracy'][-1],
            'Validation Accuracy:', metrics['validationAccuracy'][-1])
        return metrics


xTrain, y_train, xTest, y_test = embeddingLayer.getData(True)

x_train, x_test = [], []
for i in xTrain:
    embeddings = []
    for j in range(len(i)):
        embeddings.append(i[j][1])
    
    x_train.append(embeddings)

for i in xTest:
    embeddings = []
    for j in range(len(i)):
        embeddings.append(i[j][1])
    
    x_test.append(embeddings)


hidden = TreeRNN1(x_train, y_train, [311, 64, 64, 2], "relu", 0.03, 5)
hidden.runModel(x_train, y_train, x_test, y_test)
print()
