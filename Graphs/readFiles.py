import os, ast, random, dgl
import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from ASTToGraph import ASTToGraph

merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort"
quick = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Quick Sort"

def convertToGraph(filePath):
    programAST = ''
    with open (filePath, "r") as file:
        programAST = ast.parse(file.read())

    graph = ASTToGraph()
    graph.generic_visit(programAST)
    # start.createAdjList()
    graph = graph.convertToGraph()
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

def convertToDGL(x_list):
    x_dgl = []
    for graph in x_list:
        g = nx.DiGraph()
        edges = graph.edges
        for edge in edges:
            node0, node1 = np.float64(1/hash(edge[0])/255.), np.float64(1/hash(edge[1])/255.)
            nodeTypeIndicator0, nodeTypeIndicator1, edgeTypeIndicator = 1, 1, 1
            if len(list(ast.iter_child_nodes(edge[1]))) == 0:
                nodeTypeIndicator1 = 2
                edgeTypeIndicator = 2
            if g.has_node(node0) is False:
                g.add_node(node0, encoding = 1/hash(edge[0])/255., nodeType = nodeTypeIndicator0)
            if g.has_node(node1) is False:
                g.add_node(node1, encoding = 1/hash(edge[1])/255., nodeType = nodeTypeIndicator1)
            g.add_edge(node0, node1, edgeType = edgeTypeIndicator)

        x_dgl.append(dgl.from_networkx(g, node_attrs=['nodeType', 'encoding'], edge_attrs=['edgeType']))

    return x_dgl

def getParsedFiles():
    x_train, y_train, x_test, y_test = splitTrainTest()

    x_train = convertToDGL(x_train)
    y_train = tf.keras.utils.to_categorical(y_train)  

    x_test = convertToDGL(x_test)
    y_test = tf.keras.utils.to_categorical(y_test)  

    return x_train, y_train, x_test, y_test

# x_train, y_train, x_test, y_test = splitTrainTest()
# print(convertToDGL(x_train))
