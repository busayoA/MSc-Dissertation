import tensorflow as tf
import numpy as np
from typing import List
import TreeEmbeddingLayer as embeddingLayer
from TreeEmbeddingLayer import TreeEmbeddingLayer as tel
from Node import Node
from TreeRNN import TreeRNN

# USING PADDED TREES
class PaddedTreeRNN(TreeRNN):
    def __init__(self, trees: List[Node], labels, layers, activationFunction: str, learningRate: float, 
    epochs: int):
       super().__init__(trees, labels, layers, activationFunction, learningRate, epochs)

    def initialiseWeights(self):
        for i in range(1, self.layerCount):
            self.weights[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], self.layers[i-1])))
            self.bias[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], 1)))

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
                loss = self.backPropagate(tree, y_train[index])
                # SECOND FORWARD PASS/RECURRENT LOOP
                newOutputs = self.RNNLayer(tree)
                pred = tf.argmax(tf.nn.softmax(newOutputs), axis = 1)
                predictions.append(pred)
                index += 1

                metrics['trainingLoss'].append(loss)
            predictions = tf.convert_to_tensor(predictions)
            # print(predictions)
            unseenPredictions = self.makePrediction(x_test)
           
            metrics['trainingAccuracy'].append(np.mean(np.argmax(y_train, axis=1) == predictions.numpy()))
            metrics['validationAccuracy'].append(np.mean(np.argmax(y_test, axis=1) == unseenPredictions.numpy()))
            print('\tLoss:', metrics['trainingLoss'][-1], 'Training Accuracy:', metrics['trainingAccuracy'][-1],
            'Validation Accuracy:', metrics['validationAccuracy'][-1])
        return metrics
