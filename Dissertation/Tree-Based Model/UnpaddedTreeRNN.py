import tensorflow as tf
import numpy as np
from typing import List
import TreeEmbeddingLayer as embeddingLayer
from Node import Node
from TreeRNN import TreeRNN

# USING PADDED TREES
class UnpaddedTreeRNN(TreeRNN):
    def __init__(self, trees: List[Node], labels, layers, activationFunction: str, learningRate: float, 
    epochs: int):
       super().__init__(trees, labels, layers, activationFunction, learningRate, epochs)

    def initialiseWeights(self):
        for i in range(len(self.trees)):
            tree = self.trees[i]
            self.weights[i] = tf.Variable(tf.random.normal(shape=(2, 1)))
            self.bias[i] = tf.Variable(tf.random.normal(shape=(2, 1)))

    def updateWeights(self, index):
        self.weights[index].assign_sub(self.learningRate * self.weightDeltas[index])
        self.bias[index].assign_sub(self.learningRate * self.biasDeltas[index])

    def RNNLayer(self, tree, treeCount):
        weights = self.weights[treeCount]
        bias = self.bias[treeCount]
        predictions = []
        agg = []
        for t in range(len(tree)):
            currentNode = tree[t]
            outputs = (currentNode * weights) + bias
            agg.append(self.aggregationLayer("max", outputs, 1))
            predictions = self.activationFunction(agg[-1])
        return predictions

    def getAggregationFunction(self, aggregationFunction: str):
        aggregationFunction = aggregationFunction.lower()
        if aggregationFunction == "max":
            return tf.reduce_max
        else:
            return None

    def aggregationLayer(self, aggregationFunction: str, nodeEmbeddings: List, axis: int):
        # nodeEmbeddings = tf.reshape(nodeEmbeddings, (1, len(nodeEmbeddings)))
        aggregationFunction = self.getAggregationFunction(aggregationFunction)
        return aggregationFunction(nodeEmbeddings, axis=axis)

    def backPropagate(self, tree, yValues, treeCount):
        with tf.GradientTape(persistent=True) as tape: 
            output = self.RNNLayer(tree, treeCount)
            loss = self.lossFunction(output, yValues)
        
        self.weightDeltas[treeCount] = tape.gradient(loss, self.weights[treeCount])
        self.biasDeltas[treeCount] = tape.gradient(loss, self.bias[treeCount])
        
        del tape
        self.updateWeights(treeCount)
        return loss.numpy()

    def runModel(self, x_train, y_train, x_test, y_test):
        index = 0
        metrics = {'trainingLoss': [], 'trainingAccuracy': []}
        self.initialiseWeights()
        for i in range(self.epochs):
            predictions = []
            if index >= len(x_train):
                index = 0
            print('Epoch {}'.format(i+1), end='........')
            for tree in x_train:
                # FIRST FORWARD PASS
                if index % 10 == 0:
                    print(end=".")
                self.backPropagate(tree, y_train[index], index)
                newOutputs = self.RNNLayer(tree, index)
                pred = tf.argmax(tf.nn.softmax(newOutputs))
                predictions.append(pred)
                index += 1
            predictions = tf.convert_to_tensor(predictions)
            metrics['trainingAccuracy'].append(np.mean(np.argmax(y_train, axis=1) == predictions.numpy()))
            print('Accuracy:', metrics['trainingAccuracy'][-1])
        return metrics

