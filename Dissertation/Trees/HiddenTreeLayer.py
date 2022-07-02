import tensorflow as tf
import numpy as np
from typing import List
from TreeInputLayer import Node 

class HiddenTreeLayer:
    def __init__(self, trees: List[Node], labels, layers, activationFunction: str, learningRate: float, epochs: int):
        self.trees = trees
        self.labels = labels
        self.layers = layers
        self.treeCount = len(trees)
        self.layerCount = len(self.layers)
        self.hiddenCount = self.layerCount-2
        self.activationFunction = self.getActivationFunction(activationFunction)
        self.learningRate = learningRate
        self.epochs = epochs
        self.weights, self.bias, self.weightDeltas, self.biasDeltas = {}, {}, {}, {}

    def initialiseWeights(self):
        for i in range(self.treeCount):
            tree = self.trees[i]
            fullTree = tree.preorderTraversal(tree)
            self.weights[i] = tf.Variable(tf.random.normal(shape=(len(fullTree), 1)))
            self.bias[i] = tf.Variable(tf.random.normal(shape=(len(fullTree), 1)))

    def getDirectChildren(self, root: Node):
        childObjects = root.children
        childEmbeddings = []
        for i in range(len(childObjects)):
            childEmbeddings.append(childObjects[i].embedding)

        return childObjects, childEmbeddings

    def prepareEmbeddingsOnOneTree(self):
        """ Test the model on one tree """
        self.tester = self.trees[0] 
        self.testerTreeEmbeddings, self.testerTreeObjects = self.trees[0].preorderTraversal(self.trees[0])
        nodeCount = len(self.testerTreeEmbeddings)

        for i in range(nodeCount):
            currentNode = self.testerTreeObjects[i]
            children = self.getDirectChildren(currentNode)[1]
            if len(children) > 0:
                for j in range(len(children)):
                    workingTensor = [self.testerTreeEmbeddings[i], children[j]]
                    x = tf.math.reduce_mean(tf.math.log(workingTensor))
                    self.testerTreeEmbeddings[i] = x
            else:
                self.testerTreeEmbeddings[i] = tf.convert_to_tensor(self.testerTreeEmbeddings[i])

        self.testerTreeEmbeddings = tf.convert_to_tensor(self.testerTreeEmbeddings)
        self.testerTreeEmbeddings = tf.reshape(self.testerTreeEmbeddings, (1, len(self.testerTreeEmbeddings)))

        return self.testerTreeEmbeddings

    # def RNNLayer(self, embeddings, yValues, activationFunction):
    #     embeddings = tf.expand_dims(embeddings, axis=2)
    #     extension = tf.constant(np.asarray([i*np.ones(embeddings.shape[1]) for i in range(0, embeddings.shape[0])], dtype=np.float32), dtype=tf.float32)
    #     extension = tf.expand_dims(extension, axis=2)
    #     embeddings = tf.concat([extension, embeddings], axis=2)
    #     activationFunction = self.getActivationFunction(self.activationFunction)
    #     #FIRST HIDDEN LAYER
    #     rnnCell = tf.keras.layers.SimpleRNNCell(2, activation=activationFunction)
    #     rnn = tf.keras.layers.RNN(rnnCell,return_sequences=True, return_state=True)
    #     output, states = rnn(embeddings)
    #     return output, states

    def aggregationLayer(self, aggregationFunction: str, nodeEmbeddings: List, axis: int):
        # nodeEmbeddings = tf.reshape(nodeEmbeddings, (1, len(nodeEmbeddings)))
        aggregationFunction = self.getAggregationFunction(aggregationFunction)
        return aggregationFunction(nodeEmbeddings, axis=axis)

    def testModelOnOneTree(self, treeEmbeddings, aggregationFunction):
        ffnCell = self.FFNCell(treeEmbeddings, 0, aggregationFunction)
        return ffnCell
            
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

    def getAggregationFunction(self, aggregationFunction: str):
        aggregationFunction = aggregationFunction.lower()
        if aggregationFunction == "max":
            return tf.reduce_max
        else:
            return None

    def traverseTreeBranch(self, root: Node):
        return self.getDirectChildren(root)

    def prepareEmbeddings(self):
        allEmbeddings = []
        for i in range(self.treeCount):
            treeEmbeddings, treeObjects = self.trees[i].preorderTraversal(self.trees[i])
            nodeCount = len(treeEmbeddings)
            for j in range(nodeCount):
                currentNode = treeObjects[j]
                children = self.getDirectChildren(currentNode)[1]
                if len(children) > 0:
                    for k in range(len(children)):
                        workingTensor = [treeEmbeddings[j], children[k]]
                        treeEmbeddings[j] = tf.math.reduce_mean(tf.math.log(workingTensor))
                else:
                    treeEmbeddings[j] = tf.convert_to_tensor(treeEmbeddings[j])
            treeEmbeddings = tf.convert_to_tensor(treeEmbeddings)
            allEmbeddings.append(treeEmbeddings)
        return allEmbeddings

    def FFNCell(self, fullTree, treeCount, aggregationFunction: str):
        weights = self.weights[treeCount]
        bias = self.bias[treeCount]
        agg = []
        for t in range(len(fullTree)):
            currentNode = fullTree[t]
            outputs = (currentNode * weights) + bias
            agg.append(self.aggregationLayer(aggregationFunction, outputs, 1))
        predictions = self.activationFunction(agg[-1])

        return predictions

    def backPropagate(self, tree, treeCount, aggregationFunction, yValues):
        with tf.GradientTape(persistent=True) as tape:
            outputs = self.FFNCell(tree, treeCount, aggregationFunction)
            loss = self.lossFunction(outputs, yValues)
        self.weightDeltas[treeCount] = tape.gradient(loss, self.weights[treeCount])
        self.biasDeltas[treeCount] = tape.gradient(loss, self.bias[treeCount])
            # print(self.weightErrors[i])
            # print(self.biasErrors[i])
        del tape
        self.updateWeights(treeCount)

    def updateWeights(self, treeCount):
        self.weights[treeCount].assign_sub(self.learningRate * self.weightDeltas[treeCount])
        self.bias[treeCount].assign_sub(self.learningRate * self.biasDeltas[treeCount])

    def lossFunction(self, outputs, yValues):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yValues, outputs))

    def runModel(self, xTrain, yTrain, aggregationFunction):
        print("Without Back Propagation")
        index = 0
        metrics = {'trainingAccuracy': []}
        self.initialiseWeights()
        for i in range(self.epochs):
            predictions = []
            if index >= len(xTrain):
                index = 0
            print('Epoch {}'.format(i), end='........')
            for tree in xTrain:
                outputs = self.FFNCell(tree, index, aggregationFunction)
                pred = tf.argmax(tf.nn.softmax(outputs))
                predictions.append(pred)
                index += 1
            predictions = tf.convert_to_tensor(predictions)
            metrics['trainingAccuracy'].append(np.mean(np.argmax(yTrain, axis=1) == predictions.numpy()))
            print('Accuracy:', metrics['trainingAccuracy'][-1])
        return metrics

    def runModelWithBackPropagation(self, xTrain, yTrain, aggregationFunction):
        print("Applying Back Propagation")
        index = 0
        metrics = {'trainingLoss': [], 'trainingAccuracy': []}
        self.initialiseWeights()
        for i in range(self.epochs):
            predictions = []
            if index >= len(xTrain):
                index = 0
            print('Epoch {}'.format(i), end='........')
            for tree in xTrain:
                # FIRST FORRWARD PASS
                self.backPropagate(tree, index, aggregationFunction, yTrain[index])
                newOutputs = self.FFNCell(tree, index, aggregationFunction)
                pred = tf.argmax(tf.nn.softmax(newOutputs))
                predictions.append(pred)
                index += 1
            predictions = tf.convert_to_tensor(predictions)
            metrics['trainingAccuracy'].append(np.mean(np.argmax(yTrain, axis=1) == predictions.numpy()))
            print('Accuracy:', metrics['trainingAccuracy'][-1])
        return metrics


