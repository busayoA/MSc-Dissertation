import random
import tensorflow as tf
import numpy as np
from typing import List
from Node import Node
from AbstractTree import AbstractTree

class TreeRNN(AbstractTree):
    def __init__(self, trees: List[Node], labels, layers, activationFunction: str, learningRate: float, 
    epochs: int):
       super().__init__(trees, labels, layers, activationFunction, learningRate, epochs)

    def initialiseWeights(self):
        # input Layer

        for i in range(1, self.layerCount):
            self.weights[i] = tf.Variable(tf.random.normal(shape=(self.layers[i-1], 1)))
            if i == self.layerCount - 1:
                self.weights[i] = tf.Variable(tf.random.normal(shape=(2, 64)))
            self.bias[i] = tf.Variable(tf.random.normal(shape=(1, 1)))

    def updateWeights(self):
        for i in range(1, self.layerCount):
            if self.weightDeltas[i] is not None:
                self.weights[i].assign_sub(self.learningRate * self.weightDeltas[i])

            if self.biasDeltas[i] is not None:
                self.bias[i].assign_sub(self.learningRate * self.biasDeltas[i])

    def RNNLayer(self, tree):
        for i in range(1, self.layerCount): 
            weights = self.weights[i]
            if i == self.layerCount - 1:
                weights = tf.transpose(self.weights[i])
            if i == 1:
                tree = (tree * weights) + self.bias[i]
            else:
                tree = tf.matmul(tree.numpy(), tf.transpose(self.weights[i])) + self.bias[i]
            tree = self.activationFunction(tree)
        return tree

    def getAggregationFunction(self, aggregationFunction: str):
        aggregationFunction = aggregationFunction.lower()
        if aggregationFunction == "max":
            return tf.reduce_max
        elif aggregationFunction == "logsumexp":
            return tf.reduce_mean
        elif aggregationFunction == "mean":
            return tf.reduce_mean
        elif aggregationFunction == "min":
            return tf.reduce_min
        elif aggregationFunction == "prod":
            return tf.reduce_prod
        elif aggregationFunction == "sum":
            return tf.reduce_sum
        elif aggregationFunction == "std":
            return tf.math.reduce_std
        elif aggregationFunction == "var":
            return tf.math.reduce_variance
        else:
            return None

    def segmentationFunction(self, segmentationFunction: str):
        segmentationFunction = segmentationFunction.split("_")
        if segmentationFunction[0] == "unsorted":
            if segmentationFunction[1] == "sum":
                return tf.math.segment_sum
            if segmentationFunction[1] == "mean":
                return tf.math.segment_mean
            if segmentationFunction[1] == "max":
                return tf.math.segment_max
            if segmentationFunction[1] == "min":
                return tf.math.segment_min
            if segmentationFunction[1] == "prod":
                return tf.math.segment_prod
        elif segmentationFunction[0] == "sorted":
            if segmentationFunction[1] == "sum":
                return tf.math.unsorted_segment_sum
            if segmentationFunction[1] == "mean":
                return tf.math.unsorted_segment_mean
            if segmentationFunction[1] == "max":
                return tf.math.unsorted_segment_max
            if segmentationFunction[1] == "min":
                return tf.math.unsorted_segment_min
            if segmentationFunction[1] == "prod":
                return tf.math.unsorted_segment_min
        else:
            return None

    def aggregationLayer(self, aggregationFunction: str, nodeEmbeddings: List, axis: int):
        # nodeEmbeddings = tf.reshape(nodeEmbeddings, (1, len(nodeEmbeddings)))
        aggregationFunction = self.getAggregationFunction(aggregationFunction)
        return aggregationFunction(nodeEmbeddings, axis=axis)

    def segmentationLayer(self, segmentationFunction: str, nodeEmbeddings: List):
        segmentationFunction = self.segmentationFunction(segmentationFunction)
        return segmentationFunction(nodeEmbeddings, tf.constant([0, 1, 2, 4, 5, 6, 7, 8, 9]))

    def backPropagate(self, tree, yValues):
        with tf.GradientTape(persistent=True) as tape: 
            output = self.RNNLayer(tree)
            loss = self.lossFunction(output, yValues)
        
        for i in range(1, self.layerCount):
            tape.watch(self.weights[i])
            self.weightDeltas[i] = tape.gradient(loss, self.weights[i])
            self.biasDeltas[i] = tape.gradient(loss, self.bias[i])
        
        del tape
        self.updateWeights()
        return loss.numpy()

    def runModel(self, x_train, y_train, x_test, y_test):
        index = 0
        loss = 0.0
        metrics = {'trainingLoss': [], 'trainingAccuracy': [], 'validationAccuracy': []}
        self.initialiseWeights()
        for i in range(self.epochs):
            predictions = []
            for tree in x_train:
                if index % 5 == 0:
                    print(end=".")
                if index >= len(y_train):
                    index = 0

                # FIRST FORWARD PASS
                loss = self.backPropagate(tree, y_train[index])
                # SECOND FORWARD PASS/RECURRENT LOOP
                newOutputs = self.RNNLayer(tree)
                newOutputs = tf.reshape(newOutputs, (2, 1))
                pred = tf.argmax(tf.nn.softmax(newOutputs), axis = 1)
                predictions.append(pred)
                index += 1

            predictions = tf.convert_to_tensor(predictions)
            print(predictions)
            metrics['trainingLoss'].append(loss)
            unseenPredictions = self.makePrediction(x_test)
            metrics['trainingAccuracy'].append(np.mean(np.argmax(y_train, axis=1) == predictions.numpy()))
            metrics['validationAccuracy'].append(np.mean(np.argmax(y_test, axis=1) == unseenPredictions.numpy()))
            print('\Loss:', metrics['trainingLoss'][-1], 'Accuracy:', metrics['trainingAccuracy'][-1], 
            'Validation Accuracy:', metrics['validationAccuracy'][-1])
        return metrics

