import random
import tensorflow as tf
from typing import List

class TreeEmbeddingLayer():
    def __init__(self, values: list[List, List]):
        self.root = values[0]
        self.nodes =  self.root.preOrderTraversal(self.root)
        self.label = values[1]
        self.rootVec = random.random()
        self.weights = {}
        self.bias = {}
        self.vectorEmbeddings = [[self.root, self.rootVec]]
        self.vectors = [self.rootVec]
        self.treeDepth = self.getTreeDepth(self.root)
        self.unVectorised = self.root.preOrderTraversal(self.root)
        self.rootIndex = self.nodes.index(self.root)
        self.unVectorised.remove(self.root)
       
        self.initialiseInputWeights()
        self.embeddingFunction(self.root, None)

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
    
    def embeddingFunction(self, node, parent):
        if len(self.unVectorised) == 0:
            return self.vectors
        
        if parent is None:
            functionNodes = node.children
            for function in functionNodes:
                if function in self.unVectorised:
                    self.unVectorised.remove(function)
                    rootIndex = self.nodes.index(self.root)
                    functionIndex = self.nodes.index(function)
                    vec = self.vecFunction(len(functionNodes), rootIndex, function, functionIndex)
                    self.vectors.append(vec)
                    self.vectorEmbeddings.append([function, vec])
                    for child in function.children:
                        self.embeddingFunction(child, function)
        else:
            if node in self.unVectorised:
                self.unVectorised.remove(node)
                parentIndex = self.nodes.index(parent)
                childIndex = self.nodes.index(node)
                vec = self.vecFunction(len(parent.children), parentIndex, node, childIndex)
                self.vectors.append(vec)
                self.vectorEmbeddings.append([node, vec])
                for child in node.children:
                    self.embeddingFunction(child, node)
        
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
        if result < 0:
            result = result * -1.0
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


