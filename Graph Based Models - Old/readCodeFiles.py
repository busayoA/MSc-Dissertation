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

    x_train, x_train_graph = prepareGraphs(x_train_graph)
    x_test, x_test_graph = prepareGraphs(x_test_graph)

    return x_train, y_train, x_test, y_test, x_train_graph, x_test_graph, 

def prepareGraphs(xGraph):
    totalGraph, totalList = [], []
    for graph in xGraph:
        G = nx.DiGraph()
        edges = graph.edges
        for edge in edges:
            nodeTypeIndicator0 = 1
            nodeTypeIndicator1 = 1
            edgeTypeIndicator = 1
            node0 = edge[0]
            node1 = edge[1]
            if len(list(ast.iter_child_nodes(node1))) == 0:
                nodeTypeIndicator1 = 2
                edgeTypeIndicator = 2
            className0 = edge[0].__class__.__name__
            className1 = edge[1].__class__.__name__
            
            if G.has_node(edge[0]) is False:
                G.add_node(edge[0], encoding = 1/hash(edge[0])/255., className = className0, nodeType = nodeTypeIndicator0)
            if G.has_node(edge[1]) is False:
                G.add_node(edge[1], encoding = 1/hash(edge[1])/255., className = className1, nodeType = nodeTypeIndicator1)
            G.add_edge(edge[0], edge[1], hashed = [1/hash(edge[0])/255., 1/hash(edge[1])/255.], edgeType = edgeTypeIndicator)
        xList = nx.to_numpy_array(G)
        totalGraph.append(G)
        totalList.append(xList)
    totalList = np.array(totalList, dtype=object)
    return totalList, totalGraph

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

# x_train, y_train, x_test, y_test, x_train_graph, x_test_graph = readCodeFiles()
# # getGraphDetails(x_train_graph)
# prepareGraphs(x_train_graph, y_train)


# x_train_graph, x_train_array, y_train, x_test_graph, x_test_array, y_test = readCodeFiles()
# print(np.asarray(x_train_array).shape)

# x_train_graph, x_train_list, y_train, x_test_graph, x_test_list, y_test  = readCodeFiles()
# visualize(x_train_graph)

# asArray = asArray/255.
# asMatrix = asMatrix/255.
# print(asMatrix)

