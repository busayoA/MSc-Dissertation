import tensorflow as tf
import numpy as np
from typing import List
from TreeInputLayer import Node, TreeInputLayer

class HiddenTreeLayer:
    def __init__(self, trees: List[Node], labels, layers, activationFunction: str, learningRate: float, 
    epochs: int, aggregationFunction: str):
        self.trees = trees
        self.labels = labels
        self.layers = layers
        self.layerCount = len(self.layers)
        self.treeCount = len(self.trees)
        self.activationFunction = self.getActivationFunction(activationFunction)
        self.aggregationFunction = aggregationFunction
        self.learningRate = learningRate
        self.epochs = epochs
        self.weights, self.bias, self.weightDeltas, self.biasDeltas = {}, {}, {}, {}

    def initialiseWeights(self):
        for i in range(1, self.layerCount):
            self.weights[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], self.layers[i-1])))
            self.bias[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], 1)))

    def getDirectChildren(self, root: Node):
        childObjects = root.children
        childEmbeddings = []
        for i in range(len(childObjects)):
            childEmbeddings.append(childObjects[i].embedding)

        return childObjects, childEmbeddings

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

    def prepareEmbeddingsOnOneTree(self):
        """ Test the model on one tree """
        self.tester = self.trees[0] 
        self.testerTreeEmbeddings = self.treeNodes[0]
        self.testerTreeObjects = self.tester.preorderTraversal(self.tester)[1]
        nodeCount = len(self.testerTreeObjects)
        
        for i in range(nodeCount):
            currentNode = self.testerTreeObjects[i]
            children = self.getDirectChildren(currentNode)[1]
            if len(children) > 0:
                for j in range(len(children)):
                    workingTensor = [self.testerTreeEmbeddings[i], children[j]]
                    agg = self.aggregationLayer(self.aggregationFunction, workingTensor)
                    x = tf.math.reduce_mean(tf.math.log(agg))
                    self.testerTreeEmbeddings[i] = x
            else:
                self.testerTreeEmbeddings[i] = tf.convert_to_tensor(self.testerTreeEmbeddings[i])

        self.testerTreeEmbeddings = tf.convert_to_tensor(self.testerTreeEmbeddings)
        self.testerTreeEmbeddings = tf.reshape(self.testerTreeEmbeddings, (1, len(self.testerTreeEmbeddings)))

        return self.testerTreeEmbeddings

    def testModelOnOneTree(self, treeEmbeddings, aggregationFunction):
        self.initialiseWeights()
        ffnCell = self.FFNCell(treeEmbeddings, aggregationFunction)
        return ffnCell

    def FFNCell(self, tree):
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
            output = self.FFNCell(tree)
            loss = self.lossFunction(output, yValues)
        
        for i in range(1, self.layerCount):
            self.weightDeltas[i] = tape.gradient(loss, self.weights[i])
            self.biasDeltas[i] = tape.gradient(loss, self.bias[i])
        del tape
        self.updateWeights()
        return loss.numpy()

    def aggregationLayer(self, aggregationFunction: str, nodeEmbeddings: List):
        # nodeEmbeddings = tf.reshape(nodeEmbeddings, (1, len(nodeEmbeddings)))
        aggregationFunction = self.getAggregationFunction(aggregationFunction)
        return aggregationFunction(nodeEmbeddings)

    def updateWeights(self):
        for i in range(1, self.layerCount):
            self.weights[i].assign_sub(self.learningRate * self.weightDeltas[i])
            self.bias[i].assign_sub(self.learningRate * self.biasDeltas[i])

    def lossFunction(self, outputs, yValues):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yValues, outputs))

    def prepareAllEmbeddings(self, treeNodes):
        allEmbeddings = treeNodes
        embeddings = []
        for i in range(len(allEmbeddings)):
            treeObjects = self.trees[i].preorderTraversal(self.trees[i])[1]
            treeEmbeddings = allEmbeddings[i]
            for j in range(len(treeObjects)):
                currentNode = treeObjects[j]
                children = self.getDirectChildren(currentNode)[1]
                if len(children) > 0:
                    for k in range(len(children)):
                        workingTensor = [treeEmbeddings[j], children[k]]
                        agg = self.aggregationLayer(self.aggregationFunction, workingTensor)
                        treeEmbeddings[j] = tf.math.reduce_mean(tf.math.log(agg))
                else:
                    treeEmbeddings[j] = tf.convert_to_tensor(treeEmbeddings[j])
            treeEmbeddings = tf.convert_to_tensor(treeEmbeddings)
            embeddings.append(treeEmbeddings)
        return embeddings

    def makePrediction(self, x_test):
        prediction = []
        predictions = []
        for tree in x_test:
            output = self.FFNCell(tree)
            prediction = tf.argmax(tf.nn.softmax(output), axis=1)
            predictions.append(prediction)
        return tf.convert_to_tensor(predictions)

    def runModel(self, x_train, y_train, x_test, y_test):
        print("Without Back Propagation")
        index = 0
        metrics = {'trainingAccuracy': [], 'validationAccuracy': []}
        self.initialiseWeights()
        for i in range(self.epochs):
            predictions = []
            print('Epoch {}'.format(i), end='.')
            for tree in x_train:
                if index % 5 == 0:
                    print(end=".")
                if index >= len(y_train):
                    index = 0
                outputs = self.FFNCell(tree)
                pred = tf.argmax(tf.nn.softmax(outputs), axis = 1)
                predictions.append(pred)
                index += 1
            predictions = tf.convert_to_tensor(predictions)
            # print(predictions)
            unseenPredictions = self.makePrediction(x_test)
            metrics['trainingAccuracy'].append(np.mean(np.argmax(y_train, axis=1) == predictions.numpy()))
            metrics['validationAccuracy'].append(np.mean(np.argmax(y_test, axis=1) == unseenPredictions.numpy()))
            print('\tTraining Accuracy:', metrics['trainingAccuracy'][-1],'Validation Accuracy:', metrics['validationAccuracy'][-1])

    def runModelWithBackPropagation(self, x_train, y_train, x_test, y_test):
        print("Applying Back Propagation")
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
                newOutputs = self.FFNCell(tree)
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
