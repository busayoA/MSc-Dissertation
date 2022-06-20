import os, ast
import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from Visitor import Visitor

merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort"
quick = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Quick Sort"
other = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Other"

def convertToGraph(filePath):
    programAST = ''
    with open (filePath, "r") as file:
        programAST = ast.parse(file.read())

    visitor = Visitor()
    visitor.generic_visit(programAST)
    # start.createAdjList()
    graph = visitor.convertToGraph()
    return graph

def assignLabels(filePath):
    graphs, labels = [], []
    os.chdir(filePath)
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".py"):
            path = f"{filePath}/{file}"
            # call read text file function
            graphData = convertToGraph(path)
            graphs.append(graphData)
            if filePath.find("Merge") != -1:
                labels.append(0)
            elif filePath.find("Quick") != -1:
                labels.append(1)

    return graphs, labels

def splitTrainTest():
    mGraphs, mLabels = assignLabels(merge)
    qGraphs, qLabels = assignLabels(quick)

    mSplit = int(0.6 * len(mGraphs))
    qSplit = int(0.6 * len(qGraphs))

    x_train = mGraphs[:mSplit] + qGraphs[:qSplit]
    y_train = mLabels[:mSplit] + qLabels[:qSplit]
    
    x_test = mGraphs[mSplit:] + qGraphs[qSplit:]
    y_test = mLabels[mSplit:] + qLabels[qSplit:]
    

    return x_train, y_train, x_test, y_test

def getEdges(x_list):
    edgeList = []
    for graph in x_list[:1]:
        currentEdges = []
        edges = graph.edges
        # print(edges)
        for edge in edges:
            node0, node1 = np.float32(1/hash(edge[0])/255.), np.float32(1/hash(edge[1])/255.)
            currentEdges.append([node0, node1])
        edgeList.append(currentEdges)
    
    return edgeList

def convertToMatrix(x_list):
    graphs = []
    for graph in x_list:
        graph = nx.to_numpy_array(graph)
        graph = tf.convert_to_tensor(graph, dtype=np.float32)
        graphs.append(graph)

    return graphs

def getParsedFiles():
    x_train, y_train, x_test, y_test = splitTrainTest()

    x_train_matrix = convertToMatrix(x_train)
    y_train = tf.keras.utils.to_categorical(y_train)  

    x_test_matrix = convertToMatrix(x_test)
    y_test = tf.keras.utils.to_categorical(y_test)  

    return x_train, x_train_matrix, y_train, x_test, x_test_matrix, y_test

        

        
    # embeddings = []
    # for i in range(len(xList[0])):
    #     nodeEmbeddings = []
    #     graph = xList[i]
    #     nodes = graph.nodes
    #     nodes = tf.convert_to_tensor(nodes, dtype=np.float32)
    #     for j in range(len(xList)):
    #         for node in nodes:
    #             node = tf.Variable(node)
    #             node.assign(sum(nodes[j] * matrix[i][j]))
    #             nodeEmbeddings.append(node)
        
    #         nodeEmbeddings = tf.convert_to_tensor(nodeEmbeddings, dtype=np.float32)
    #     embeddings.append(nodeEmbeddings)
    
    # embeddings = tf.convert_to_tensor(embeddings, dtype=np.float32)
    # return embeddings

    # # INPUT LAYER
    # nodes = x_train[0].ndata['encoding']
    # nodes = tf.convert_to_tensor(nodes, dtype=np.float32)
    # # for node in nodes:
    # #     count = 0
    # #     for i in range(len(x_train_edges)):
    # #         for j in range(1):
    # #             if x_train_edges[i][0] == node:
    # #                 pass
    # labels = x_train[1]
    # adjList = tf.convert_to_tensor(x_train_matrix, dtype=np.float32)
    # x = []

    # # Get the node embeddings as a function of the adjacency matrix on each node
    # for i in range(len(nodes)):
    #     node = tf.Variable(nodes[i])
    #     embedding = sum(nodes[i] * adjList)
    #     node.assign(embedding)
    #     x.append(node)

    # x = tf.convert_to_tensor(x, dtype=np.float32) 
    # x = tf.reshape(x, (1, len(x)))
    # self.layers[0] = len(nodes)

# getNodeEmbeddings(x_train, x_train_matrix)