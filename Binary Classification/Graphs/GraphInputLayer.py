import os, ast
import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from Visitor import Visitor
from abc import ABC, abstractmethod

class GraphInputLayer(ABC):
    def convertToGraph(self, filePath):
        programAST = ''
        with open (filePath, "r") as file:
            programAST = ast.parse(file.read())
        visitor = Visitor()
        visitor.generic_visit(programAST)
        # start.createAdjList()
        graph = visitor.convertToGraph()
        return graph

    def assignLabels(self, filePath):
        graphs, labels = [], []
        os.chdir(filePath)
        for file in os.listdir():
            # Check whether file is in text format or not
            if file.endswith(".py"):
                path = f"{filePath}/{file}"
                # call read text file function
                graphData = self.convertToGraph(path)
                graphs.append(graphData)
                if filePath.find("Merge") != -1:
                    labels.append(0)
                elif filePath.find("Quick") != -1:
                    labels.append(1)
                elif filePath.find("Other") != -1:
                    labels.append(2)
        return graphs, labels

    @abstractmethod
    def splitTrainTest(self, file1, file2, file3=None):
        raise NotImplementedError()

    def convertToMatrix(self, x_list):
        graphs = []
        for graph in x_list:
            graph = nx.to_numpy_array(graph)
            graph = tf.convert_to_tensor(graph, dtype=np.float32)
            graphs.append(graph)
        return graphs

    def getDatasets(self, x_train, y_train, x_test, y_test):
        x_train_matrix = self.convertToMatrix(x_train)
        y_train = tf.keras.utils.to_categorical(y_train)  

        x_test_matrix = self.convertToMatrix(x_test)
        y_test = tf.keras.utils.to_categorical(y_test)  

        return x_train, x_train_matrix, y_train, x_test, x_test_matrix, y_test

    def getAdjacencyLists(self, x_train, x_matrix):
        embeddings = list(x_train.nodes)
        adjList = nx.dfs_successors(x_train)
        adjacencies = []
        for i in range(len(embeddings)):
            node = embeddings[i]
            matrix = x_matrix[i]
            x = sum(node * matrix)
            embeddings[i] = x
            embeddings[i] = tf.convert_to_tensor(embeddings[i], dtype=np.float32)  

            for item in adjList:
                if node == item:
                    adjacentNodes = tf.convert_to_tensor(adjList[item], dtype=np.float32)
                    adjacencies.append(adjacentNodes)
                    
        x = tf.reshape(embeddings, (1, len(embeddings)))
        
        return embeddings, adjacencies
            

            
    