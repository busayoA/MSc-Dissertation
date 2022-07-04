import random
import tensorflow as tf
import parseFiles as pf
from Node import Node
from typing import List

class TreeEmbeddingLayer1():
    def __init__(self, values: list[List, List]):
        self.nodes = values[0]
        self.root = self.getRootNode()
        self.label = values[1]
        self.rootVec = random.random()
        self.weights = {}
        self.bias = {}
        self.fullTree = self.root.preOrderTraversal(self.root)
        self.vectorEmbeddings = [[self.root, self.rootVec]]
        self.treeDepth = self.getTreeDepth(self.root)
       
        self.initialiseInputWeights()
        self.embeddingFunction(self.root)
        self.embedPaddedNodes()

    def getTreeDepth(self, root):
        if root is None:
            return 0
        maxDepth = 0
        for child in root.children:
            maxDepth = max(maxDepth, self.getTreeDepth(child))   
        return maxDepth + 1

    def initialiseInputWeights(self):
        for i in range(len(self.nodes)):
            self.weights[i] = tf.Variable(tf.random.normal(shape=(1, 1)))
            self.bias[i] = tf.Variable(tf.random.normal(shape=(1, 1)))

    def getRootNode(self):
        for node in self.nodes:
            parent = self.getParentNode(node)
            if parent is None and len(node.children) > 0:
                return node
    
    def embeddingFunction(self, root):
        if len(root.children) > 0:
            for child in root.children:
                rootChildCount = len(self.root.children)
                parentIndex = self.nodes.index(root)
                childIndex =  self.nodes.index(child)
                vec = self.vecFunction(rootChildCount, parentIndex, child, childIndex)
                self.vectorEmbeddings.append([child, vec])
                self.embeddingFunction(child)

    def embedPaddedNodes(self):
        if len(self.vectorEmbeddings) < 311:
            for i in range(311-len(self.vectorEmbeddings)):
                self.vectorEmbeddings.append([Node(), 0.0])
        elif len(self.vectorEmbeddings) > 311:
            for i in range(len(self.vectorEmbeddings)-311):
                self.vectorEmbeddings.pop()

    def vecFunction(self, parentChildCount, parentIndex, child, index):
        # my function is ___________
        childCount = len(child.children)
        pre = 0.0
        if childCount > 0:
            pre = float(self.treeDepth) * (parentChildCount/childCount) *  (self.weights[parentIndex] + self.weights[index])
        else:
            pre = float(self.treeDepth) *  (self.weights[parentIndex] + self.weights[index])
        a = pre + self.bias[index]
        result = tf.reduce_logsumexp(a) * 0.1
        return result.numpy()

    def findNodeEmbedding(self, node):
        count, embedding = 0, 0.0
        for i in self.vectorEmbeddings:
            n = i[0]
            e = i[1]
            if n == node:
                embedding = e
            count += 1        
        return embedding

    def getParentNode(self, child):
        index = 0
        parent = None
        found = False
        while found is False and index < 311:
            found = True
            currentNode = self.nodes[index]
            currentNodeChildren = currentNode.children
            if child in currentNodeChildren:
                found = True
                parent = currentNode
            else:
                found = False
            index += 1
        return parent


def getData(padding: bool):
    x, y = pf.x, pf.y
    if padding is True:
        x = pf.padTrees(x)
    pairs = pf.attachLabels(x, y)
    split = int(0.8 * len(pairs))
    train, test = pairs[:split], pairs[split:]

    print("Collecting training data", end="......")
    x_train, y_train = [], []
    for i in range(len(train)):
        if i % 5 == 0:
            print(end=".")
        embedding = TreeEmbeddingLayer1(train[i])
        x_train.append(embedding.vectorEmbeddings)
        y_train.append(embedding.label)

    print("\nCollecting testing data", end="......")
    x_test, y_test = [], []
    for i in range(len(test)):
        if i % 5 == 0:
            print(end=".")
        embedding = TreeEmbeddingLayer1(test[i])
        x_test.append(embedding.vectorEmbeddings)
        y_test.append(embedding.label)
    print()

    return x_train, y_train, x_test, y_test

        