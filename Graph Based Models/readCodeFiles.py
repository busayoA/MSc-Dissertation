import os, ast, random
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from CodeToAdjList import Node

mpl.use('Qt5Agg')


merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort"
quick = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Quick Sort"

def convertToGraph(filePath):
    programAST = ''
    with open (filePath, "r") as file:
        programAST = ast.parse(file.read())

    node = Node()
    node.generic_visit(programAST)
    node.createAdjList()
    graph = node.convertToGraph()

    return node.adjList, graph

def assignLabels(filePath):
    adjLists, graphs, graphLabels = [], [], []
    os.chdir(filePath)
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".py"):
            path = f"{filePath}/{file}"
            # call read text file function
            graphData = convertToGraph(path)
            adjLists.append(graphData[0])
            graphs.append(graphData[1])
            if filePath.find("Merge") != -1:
                graphLabels.append(0)
            elif filePath.find("Quick") != -1:
                graphLabels.append(1)

    return adjLists, graphs, graphLabels

def readCodeFiles():
    mergeAdjList, mergeGraphs, mergeGraphLabels = assignLabels(merge)
    quickAdjList, quickGraphs, quickGraphLabels  = assignLabels(quick)

    x_train_graph, x_train_list, y_train, x_test_graph, x_test_list, y_test = [], [], [], []

    x_train_graph = mergeGraphs[:int(0.7*len(mergeGraphs))] + quickGraphs[:int(0.7*len(quickGraphs))]
    x_train_list = mergeAdjList[:int(0.7*len(mergeAdjList))] + quickAdjList[:int(0.7*len(quickAdjList))]
    y_train = mergeGraphLabels[:int(0.7*len(mergeGraphLabels))] + quickGraphLabels[:int(0.7*len(quickGraphLabels))] 

    x_test_graph = mergeGraphs[int(0.7*len(mergeGraphs)):] + quickGraphs[int(0.7*len(quickGraphs)):]
    x_test_list = mergeAdjList[int(0.7*len(mergeAdjList)):] + quickAdjList[int(0.7*len(quickAdjList)):]
    y_test = mergeGraphLabels[int(0.7*len(mergeGraphLabels)):] + quickGraphLabels[int(0.7*len(quickGraphLabels)):] 

    return x_train_graph, x_train_list, y_train, x_test_graph, x_test_list, y_test 

# print(x_train)

def visualize():
    """ Visaulise a random graph"""
    x_train_graph, x_train_list, y_train, x_test_graph, x_test_list, y_test = readCodeFiles()

    rand = random.randint(0, len(x_train_graph))
    G = nx.DiGraph()
    edges = x_train_graph[rand].edges
    G.add_edges_from(edges)
    print(G.edges)
    nx.draw_networkx(G)
    plt.show()

# visualize()


# def getVectorizedCodeData():
#     x_train, y_train, x_test, y_test = readCodeFiles()
#     vectorizer = CountVectorizer(tokenizer=lambda doc:doc, min_df=2)
#     vectorizer.fit(x_train)
#     x_train = vectorizer.transform(x_train)
#     x_train = x_train.toarray()/255.
#     y_train = tf.keras.utils.to_categorical(y_train)

#     x_test  = vectorizer.transform(x_test)
#     x_test = x_test.toarray()/255.
#     y_test = tf.keras.utils.to_categorical(y_test)

#     return x_train, y_train, x_test, y_test

# getVectorizedCodeData()


