import os, ast, random
import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from ASTtoGraph import ASTtoGraph



merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort"
quick = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Quick Sort"

def convertToGraph(filePath):
    programAST = ''
    with open (filePath, "r") as file:
        programAST = ast.parse(file.read())

    start = ASTtoGraph()
    start.generic_visit(programAST)
    # start.createAdjList()
    graph = start.convertToGraph()

    G = nx.DiGraph()
    G.add_edges_from(graph.edges)
    return graph

def assignLabels(filePath):
    graphs, graphLabels = [], []
    os.chdir(filePath)
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".py"):
            path = f"{filePath}/{file}"
            # call read text file function
            graphData = convertToGraph(path)
            graphs.append(graphData)
            if filePath.find("Merge") != -1:
                graphLabels.append(0)
            elif filePath.find("Quick") != -1:
                graphLabels.append(1)

    return graphs, graphLabels

def readCodeFiles():
    mergeGraphs, mergeGraphLabels = assignLabels(merge)
    quickGraphs, quickGraphLabels  = assignLabels(quick)

    x_train_graph = mergeGraphs[:int(0.7*len(mergeGraphs))] + quickGraphs[:int(0.7*len(quickGraphs))]
    y_train = mergeGraphLabels[:int(0.7*len(mergeGraphLabels))] + quickGraphLabels[:int(0.7*len(quickGraphLabels))] 
    y_train = tf.keras.utils.to_categorical(y_train)  

    x_test_graph = mergeGraphs[int(0.7*len(mergeGraphs)):] + quickGraphs[int(0.7*len(quickGraphs)):]
    y_test = mergeGraphLabels[int(0.7*len(mergeGraphLabels)):] + quickGraphLabels[int(0.7*len(quickGraphLabels)):] 
    y_test = tf.keras.utils.to_categorical(y_test)  

    gTrain = nx.DiGraph()
    index = 0
    for graph in x_train_graph:
        edges = graph.edges
        hashedEdges = []
        newEdges = ()
        yLabel = y_train[index]
        for edge in edges:
            edge0 = 1/hash(edge[0])
            edge1 = 1/hash(edge[1])
            gTrain.add_node(edge0, yValue=yLabel)
            gTrain.add_node(edge1, yValue=yLabel)
            gTrain.add_edge(edge0, edge1)
        #     newEdges = (edge0, edge1)
        #     hashedEdges.append(newEdges)
        # gTrain.add_edges_from(hashedEdges)
        index += 1
    x_train_graph = gTrain
    x_train = nx.to_numpy_array(gTrain)
    # x_train_matrix = nx.to_numpy_matrix(gTrain)
    
    gTest = nx.DiGraph()
    index = 0
    for graph in x_test_graph:
        edges = graph.edges
        hashedEdges = []
        newEdges = ()
        yLabel = y_test[index]
        for edge in edges:
            edge0 = 1/hash(edge[0])
            edge1 = 1/hash(edge[1])
            gTest.add_node(edge0, yValue=yLabel)
            gTest.add_node(edge1, yValue=yLabel)
            gTest.add_edge(edge0, edge1)
        #     newEdges = (edge0, edge1)
        #     hashedEdges.append(newEdges)
        # gTest.add_edges_from(hashedEdges)
        index += 1
    x_test_graph = gTest

    x_test = nx.to_numpy_array(gTest)
    # x_test_matrix = nx.to_numpy_matrix(gTest)

    return x_train, y_train, x_test, y_test, x_train_graph, x_test_graph

def visualize(xGraph):
    """ Visaulise a random graph"""
    rand = random.randint(0, len(xGraph))
    G = nx.DiGraph()
    G.add_edges_from(xGraph[rand].edges)
    print(G.edges)
    nx.draw_networkx(G)
    plt.show()

def getGraphDetails(xGraph):
    print(xGraph.number_of_nodes())
    print(xGraph.size())

x_train, y_train, x_test, y_test, x_train_graph, x_test_graph = readCodeFiles()
getGraphDetails(x_train_graph)
# x_train_graph, x_train_array, y_train, x_test_graph, x_test_array, y_test = readCodeFiles()
# print(np.asarray(x_train_array).shape)

# x_train_graph, x_train_list, y_train, x_test_graph, x_test_list, y_test  = readCodeFiles()
# visualize(x_train_graph)

# asArray = asArray/255.
# asMatrix = asMatrix/255.
# print(asMatrix)

