import random
import tensorflow as tf
import parseFiles as pf
from Node import Node
from typing import List
from os.path import dirname, join

class TreeEmbeddingLayer():
    def __init__(self, values: list[List, List], padding: bool):
        self.padding = padding
        self.nodes = values[0]
        self.root = self.getRootNode()
        self.fullTree = self.root.preOrderTraversal(self.root)
        self.label = values[1]
        self.rootVec = random.random()
        self.weights = {}
        self.bias = {}
        self.vectorEmbeddings = [[self.root, self.rootVec]]
        self.vectors = [self.rootVec]
        self.treeDepth = self.getTreeDepth(self.root)
       
        self.initialiseInputWeights()
        self.embeddingFunction(self.root)

        if self.padding is True:
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
                self.vectors.append(vec)
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
        if self.padding is True:
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
        else:
            while found is False and index < len(self.nodes):
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


def saveData(padding: bool):
    current_dir = dirname(__file__)
    x, y = pf.x, pf.y
    if padding is True:
        x = pf.getPaddedTrees(x)
        xTrain = join(current_dir, "./Data/x_train_padded.txt")
        yTrain = join(current_dir, "./Data/y_train_padded.txt")
        xTest = join(current_dir, "./Data/x_test_padded.txt")
        yTest = join(current_dir, "./Data/y_test_padded.txt")
    else:
        x = pf.getUnpaddedTrees(x)
        xTrain = join(current_dir, "./Data/x_train.txt")
        yTrain = join(current_dir, "./Data/y_train.txt")
        xTest = join(current_dir, "./Data/x_test.txt")
        yTest = join(current_dir, "./Data/y_test.txt")
    pairs = pf.attachLabels(x, y)
    split = int(0.8 * len(pairs))
    train, test = pairs[:split], pairs[split:]

    print("Collecting training data", end="......")
    x_train, y_train = [], []
    for i in range(len(train)):
        if i % 5 == 0:
            print(end=".")
        embedding = TreeEmbeddingLayer(train[i], padding)
        x_train.append(embedding.vectors)
        y_train.append(embedding.label)

    with open(xTrain, 'w') as writer:
        for i in x_train:
            writer.write(str(i) + "\n")

    with open(yTrain, 'w') as writer:
        for i in y_train:
            writer.write(str(i) + "\n")
    print()
    
    
    print("Collecting testing data", end="......")
    x_test, y_test = [], []
    for i in range(len(test)):
        if i % 5 == 0:
            print(end=".")
        embedding = TreeEmbeddingLayer(test[i], padding)
        x_test.append(embedding.vectors)
        y_test.append(embedding.label)
    with open(xTest, 'w') as writer:
        for i in x_test:
            writer.write(str(i) + "\n")
        
    with open(yTest, 'w') as writer:
        for i in y_test:
            writer.write(str(i) + "\n")
    print()
    
# saveData(False)
# saveData(True)

def readXFiles(filePath):
    with open(filePath, 'r') as reader:
        values = reader.readlines()
    
    for x in range(len(values)):
        values[x] = values[x].replace("[", "").replace("]", "").strip("\n")
        values[x] = values[x].split(",")
        values[x] = [float(i) for i in values[x]]

    return values

def readYFiles(filePath):
    with open(filePath, 'r') as reader:
        values = reader.readlines()
    
    for y in range(len(values)):
        values[y] = values[y].replace("[", "").replace("]", "").strip("\n")
        values[y] = values[y].split(" ")
        values[y] = [float(i) for i in values[y]]

    return values

def getData(padding):
    current_dir = dirname(__file__)
    if padding is True:
        xTrain = join(current_dir, "./Data/x_train_padded.txt")
        yTrain = join(current_dir, "./Data/y_train_padded.txt")
        xTest = join(current_dir, "./Data/x_test_padded.txt")
        yTest = join(current_dir, "./Data/y_test_padded.txt")
    else:
        xTrain = join(current_dir, "./Data/x_train.txt")
        yTrain = join(current_dir, "./Data/y_train.txt")
        xTest = join(current_dir, "./Data/x_test.txt")
        yTest = join(current_dir, "./Data/y_test.txt")

    x_train, y_train, x_test, y_test = [], [], [], []
    x_train = readXFiles(xTrain)
    y_train = readYFiles(yTrain)

    x_test = readXFiles(xTest)
    y_test = readYFiles(yTest)

    return x_train, y_train, x_test, y_test

getData(False)
        