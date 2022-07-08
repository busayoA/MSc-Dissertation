import os, ast, re, random
import numpy as np
import tensorflow as tf
import networkx as nx
from Visitor import GraphVisitor
from os.path import dirname, join

class GraphParser():
    def convertToGraph(self, filePath):
        programAST = ''
        with open (filePath, "r") as file:
            programAST = ast.parse(file.read())
        visitor = GraphVisitor()
        visitor.generic_visit(programAST)
        graph = visitor.convertToGraph()
        return graph

    def assignLabels(self, filePath):
        current_dir = dirname(__file__)
        filePath = join(current_dir, filePath)
        graphs, labels = [], []
        os.chdir(filePath)
        for file in os.listdir():
            if file.endswith(".py"):
                path = f"{filePath}/{file}"
                graphData = self.convertToGraph(path)
                graphs.append(graphData)
                if filePath.find("Merge") != -1:
                    labels.append(0)
                elif filePath.find("Quick") != -1:
                    labels.append(1)
                elif filePath.find("Other") != -1:
                    labels.append(2)
        return graphs, labels

    def assignLabelsToFiles(self, file1, file2):
        graph1, labels1 = self.assignLabels(file1)
        graph2, labels2 = self.assignLabels(file2)

        x = graph1 + graph2 
        y = labels1 + labels2
        
        return x, y

    def convertToMatrix(self, x):
        matrices = []
        for graph in x:
            matrix = nx.to_numpy_array(graph)
            # matrix = tf.convert_to_tensor(matrix, dtype=np.float32)
            matrices.append(matrix)
        return matrices

    def readFiles(self):
        current_dir = dirname(__file__)

        merge = "./Data/Merge Sort"
        quick = "./Data/Quick Sort"

        merge = join(current_dir, merge)
        quick = join(current_dir, quick)

        x_graph, y  = self.assignLabelsToFiles(merge, quick)

        matrices = self.convertToMatrix(x_graph)
        x_list = self.extractNodes(x_graph)
        x = []
        for i in range(len(x_list)):
            x.append([x_list[i], x_graph[i]])

        for i in range(len(x)):
            x[i][0].append(y[i])
        
        random.shuffle(x)
        labels = []
        for i in range(len(x)):
            labels.append(x[i][0][-1])
            x[i][0].pop()

        return x, matrices, labels

    def extractNodes(self, graphs):
        xNodes = []
        for i in range(len(graphs)):
            currentGraph = list(graphs[i].nodes)
            xNodes.append(currentGraph)
        return xNodes

    def getAdjacencyLists(self, x_graph, matrix):

        embeddings = list(x_graph.nodes)
        adjList = nx.dfs_successors(x_graph)
        adjacencies = []
        print(end=".")
        for i in range(len(embeddings)):
            node = embeddings[i]
            embeddings[i] = sum(sum((embeddings[i] * matrix)))

            for item in adjList:
                if item == node:
                    adjacencies.append([embeddings[i], node, adjList[item]])
            
        return embeddings, adjacencies

    def prepareData(self, graphs, matrices):
        exitGraph = []
        nodes = graphs
        print("Collecting node embeddings and adjacency lists")
        adj = [0] * len(graphs)
        for i in range(len(graphs)):
            g = graphs[i]
            graphs[i], adj[i] = self.getAdjacencyLists(g, matrices[i])

        index = 0
        for graph in graphs:  
            finalWorkingGraph = []
            adjacencies = adj[index]
            originalNodes = nodes[index]
            nodesWithAdjacentNodes = [adjacencies[i][1] for i in range(len(adjacencies))]
            adjacentNodes = [adjacencies[i][2] for i in range(len(adjacencies))]

            for i in range(len(graph)):
                currentNode = originalNodes[i]
                if currentNode in nodesWithAdjacentNodes:
                    workingValues = [currentNode]
                    ind = nodesWithAdjacentNodes.index(currentNode)
                    adjNodes = adjacentNodes[ind]
                    for j in range(len(adjNodes)):
                        workingValues.append(adjNodes[j])
                    workingValues = tf.convert_to_tensor(workingValues, dtype=np.float32)
                    x = tf.math.reduce_mean(tf.math.log(workingValues))
                    finalWorkingGraph.append(x)
                else:
                    finalWorkingGraph.append(tf.convert_to_tensor(graph[i], dtype=np.float32))
            exitGraph.append(tf.convert_to_tensor(finalWorkingGraph))
            index += 1

        return exitGraph

