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
        
        self.activationFunction = self.getActivationFunction(activationFunction.lower())
        self.weights, self.bias, self.weightDeltas, self.biasDeltas = {}, {}, {}, {}

        # for i in range(self.layerCount):
        #     self.weights[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], self.layers[i-1])))
        #     self.bias[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], 1)))

    def testModelOnOneTree(self, aggregationFunction: str):
        """ Test the model on one tree """
        self.tester = self.trees[0] 
        self.testerTreeEmbeddings, self.testerTreeObjects = self.trees[0].preorderTraversal(self.trees[0])
        nodeCount = len(self.testerTree)

        # Create an LSTM cell for each node in the tree
        for i in range(nodeCount):
            currentNode = self.tester[i]
            currentNodeEmbedding = currentNode.embedding
            children = self.getDirectChildren(currentNode)[1]

            for j in range(len(children)):
                workingTensor = [currentNodeEmbedding, children[j]]
                workingTensor = tf.convert_to_tensor(workingTensor, dtype=np.float32)
                aggFunction = self.getAggregationFunction(aggregationFunction)
                output = aggFunction(workingTensor)
            currentNodeEmbedding = output

            lstmCell = self.LSTMCell(64, self.activationFunction, False)
            nodeOutput = lstmCell(currentNodeEmbedding)
            
        for i in range(self.layerCount):
            self.weights[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], self.layers[i-1])))
            self.bias[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], 1)))
        
    def getDirectChildren(self, root: Node):
        childObjects = root.children
        childEmbeddings = []
        for i in range(len(childObjects)):
            childEmbeddings.append(childObjects[i].embedding)

        return childObjects, childEmbeddings

    def RNNCell(self, neurons: int, activationFunction: function, useBias: bool):
        return tf.keras.layers.SimpleRNNCell(neurons, activation=activationFunction, use_bias=useBias)

    def LSTMCell(self, neurons: int, activationFunction: function, useBias: bool):
        return tf.keras.layers.LSTMCell(neurons, activation=activationFunction, use_bias=useBias)

    def GRUCell(self, neurons: int, activationFunction: function, useBias: bool):
        return tf.keras.layers.GRUCell(neurons, activation=activationFunction, use_bias=useBias)

    def getActivationFunction(self, activationFunction: str):
        if activationFunction == 'softmax':
            return tf.nn.softmax
        elif activationFunction == 'relu':
            return tf.nn.relu
        elif activationFunction == 'tanh':
            return tf.tanh
        elif activationFunction == 'logsigmoid':
            def logSigmoid(x):
                weights = tf.Variable(tf.random.normal(shape=(len(x), 2)), dtype=np.float32)
                bias = tf.Variable(tf.random.normal(shape=(2, 1)), dtype=np.float32)
                x = tf.matmul(x, weights) + tf.transpose(bias)
                x = 1.0/(1.0 + tf.math.exp(-x)) 
                return x
            return logSigmoid
        else:
            return None

    def getAggregationFunction(self, aggregationFunction: str):
        aggregationFunction = aggregationFunction.lower()
        if aggregationFunction == "logsumexp":
            return tf.reduce_logsumexp
        if aggregationFunction == "max":
            return tf.reduce_max
        if aggregationFunction == "mean":
            return tf.reduce_mean
        if aggregationFunction == "prod":
            return tf.reduce_prod
        if aggregationFunction == "std":
            return tf.math.reduce_std
        if aggregationFunction == "sum":
            return tf.reduce_sum
        if aggregationFunction == "variance":
            return tf.math.reduce_variance

    # def traverseTreeBranch(self, root: Node, children):
    #     return self.getChildren(root, children)

    # def getChildren(self, root: Node, children):
    #     children = children
    #     childNodes = root.children
    #     if len(childNodes) == 0:
    #         return childNodes, children
    #     for child in childNodes:
    #         if isinstance(child, float):
    #             children.append(child)
    #         elif isinstance(child, Node):
    #             children.append(child.embedding)
    #             self.getChildren(child, children)
    #     return childNodes, children

    

    
hiddenLayer = HiddenTreeLayer()
# print(hiddenLayer.traverseTreeBranch(x_train[0], []))

root = Node(0.9)
tree, objects = root.preorderTraversal(x_train[0])

