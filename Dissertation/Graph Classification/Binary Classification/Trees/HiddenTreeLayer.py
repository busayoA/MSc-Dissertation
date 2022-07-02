import tensorflow as tf
import numpy as np
from typing import List
from BinaryTreeInputLayer import BinaryTreeInputLayer as BTIL
from BinaryTreeInputLayer import Node 

inputLayer = BTIL()
x_train, y_train, x_test, y_test = inputLayer.getData()

class HiddenTreeLayer:
    def __init__(self, trees: List[Node], labels, layers, activationFunction: str):
        self.trees = trees
        self.labels = labels
        self.layers = layers
        self.treeCount = len(trees)
        self.layerCount = len(self.layers)
        self.hiddenCount = self.layerCount-2
        self.activationFunction = activationFunction
        self.weights, self.bias, self.weightDeltas, self.biasDeltas = {}, {}, {}, {}

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

    #     rnn = tf.keras.layers.RNN(rnnCell,return_sequences=True, return_state=True )

    #     # whole_sequence_output has shape `[32, 10, 4]`.
    #     # final_state has shape `[32, 4]`.
    #     output, states = rnn(embeddings)
    #     return output, states

    def aggregationLayer(self, aggregationFunction: str, nodeEmbeddings: List):
        # nodeEmbeddings = tf.reshape(nodeEmbeddings, (1, len(nodeEmbeddings)))
        aggregationFunction = self.getAggregationFunction(aggregationFunction)
        return aggregationFunction(nodeEmbeddings, axis=1)

    def runFFNLayer(self, embeddings, activationFunction, aggregationFunction: str):
        for tree in self.trees:
            fullTree = tree.preorderTraversal(tree)
            output = self.FFNCell(fullTree, activationFunction, aggregationFunction)
        return output

    def testModelOnOneTree(self, treeEmbeddings, yValues, activationFunction, aggregationFunction):
        ffnCell = self.FFNCell(treeEmbeddings, activationFunction, 0, aggregationFunction)
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

    def FFNCell(self, fullTree, activationFunction, treeCount, aggregationFunction: str):
        weights = self.weights[treeCount]
        bias = self.bias[treeCount]
        agg = []
        for t in range(len(fullTree)):
            currentNode = fullTree[t]
            outputs = (currentNode * weights) + bias
            agg.append(self.aggregationLayer(aggregationFunction, outputs))
        predictions = activationFunction(agg[-1])

        return predictions

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

    def runModel(self, treeEmbeddings, yValues, activationFunction, aggregationFunction):
        predictions = []
        for tree in treeEmbeddings:
            ffnCell = self.FFNCell(treeEmbeddings, activationFunction, 0, aggregationFunction)
            predictions.append(ffnCell)
        return predictions

hiddenLayer = HiddenTreeLayer(x_train, y_train, [1, 158, 128, 2], "tanh")
root = Node(0.9)
tree, objects = root.preorderTraversal(x_train[0])
# x = hiddenLayer.testModelOnOneTree(tree, y_train[0], tf.nn.relu, "max")
# print(x.numpy())

y = hiddenLayer.prepareEmbeddings()
z = hiddenLayer.runModel(y, y_train, tf.nn.softmax, "max")
print(z)
tree, objects = root.preorderTraversal(x_train[0])

