import random
import tensorflow as tf
import numpy as np
import networkx as nx
from networkx import DiGraph 
from GraphParser import GraphParser

class GraphEmbeddingLayer:
    def __init__(self, graph: DiGraph):
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.nodeCopy = list(graph.nodes)
        self.edges = list(graph.edges)
        self.root = self.nodes[0]
        self.rootEmbedding = random.random()
        self.vectors = [self.rootEmbedding]
        self.nodeCopy.remove(self.root)
        self.unVectorised = self.nodeCopy

        self.weights, self.bias = {}, {}
        self.initialiseInputWeights()
        self.embeddingFunction(self.root, None)
        

    def initialiseInputWeights(self):
        for i in range(len(self.nodes)):
            self.weights[i] = tf.Variable(tf.random.normal(shape=(1, 1)))
            self.bias[i] = tf.Variable(tf.random.normal(shape=(1, 1)))


    def embeddingFunction(self, node, parent):
        if len(self.unVectorised) == 0:
            return self.vectors
        
        if parent is None:
            childNodes = nx.dfs_successors(self.graph, node)
            for child in childNodes:
                if child in self.unVectorised:
                    self.unVectorised.remove(child)
                    rootIndex = self.nodes.index(self.root)
                    childIndex = self.nodes.index(child)
                    parentChildCount = len(self.getChildNodes(self.root))
                    childNodeChildCount = len(self.getChildNodes(child))
                    vec = self.vecFunction(parentChildCount, rootIndex, childIndex, childNodeChildCount)
                    self.vectors.append(vec)
                    for childNode in self.getChildNodes(child):
                        self.embeddingFunction(childNode, child)
        else:
            if node in self.unVectorised:
                self.unVectorised.remove(node)
                parentIndex = self.nodes.index(parent)
                childIndex = self.nodes.index(node)
                parentChildCount = len(self.getChildNodes(parent))
                childNodeChildCount = len(self.getChildNodes(node))
                vec = self.vecFunction(parentChildCount, parentIndex, childIndex, childNodeChildCount)
                self.vectors.append(vec)
                for childNode in self.getChildNodes(node):
                    self.embeddingFunction(childNode, node)
    
    def getChildNodes(self, node):
        edges = self.edges
        children = []
        for edge in edges:
            if edge[0] == node:
                children.append(edge[1])

        return children
    
    def vecFunction(self, parentChildCount, parentIndex, childIndex, childNodeChildCount):
        pre = 0.0
        if childNodeChildCount > 0:
            pre = (parentChildCount/childNodeChildCount) *  (self.weights[parentIndex] + self.weights[childIndex])
        else:
            pre = (self.weights[parentIndex] + self.weights[childIndex])
        result = tf.reduce_logsumexp(pre) * 0.1
        if result < 0:
            result = result * -1.0
        return result.numpy()
