import GraphInputLayer
import os, ast
import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from Visitor import Visitor
from abc import ABC, abstractmethod

merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort"
quick = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Quick Sort"
other = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Other"

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
    def splitTrainTest(self, isOther):
        mGraphs, mLabels = self.assignLabels(merge)
        qGraphs, qLabels = self.assignLabels(quick)
        if isOther is True:
            mGraphs, mLabels = self.assignLabels(other)

        mSplit = int(0.6 * len(mGraphs))
        qSplit = int(0.6 * len(qGraphs))

        x_train = mGraphs[:mSplit] + qGraphs[:qSplit]
        y_train = mLabels[:mSplit] + qLabels[:qSplit]
        
        x_test = mGraphs[mSplit:] + qGraphs[qSplit:]
        y_test = mLabels[mSplit:] + qLabels[qSplit:]
        

        return x_train, y_train, x_test, y_test

    def convertToMatrix(self, x_list):
        graphs = []
        for graph in x_list:
            graph = nx.to_numpy_array(graph)
            graph = tf.convert_to_tensor(graph, dtype=np.float32)
            graphs.append(graph)
        return graphs

    @abstractmethod    
    def getParsedFiles(self, x_train, y_train, x_test, y_test):
        x_train_matrix = self.convertToMatrix(x_train)
        y_train = tf.keras.utils.to_categorical(y_train)  

        x_test_matrix = self.convertToMatrix(x_test)
        y_test = tf.keras.utils.to_categorical(y_test)  

        return x_train, x_train_matrix, y_train, x_test, x_test_matrix, y_test

            