import os, ast, re
import numpy as np
import tensorflow as tf
import networkx as nx
from os.path import dirname, join

class GraphInputLayer():
    def convertToGraph(self, filePath):
        programAST = ''
        with open (filePath, "r") as file:
            programAST = ast.parse(file.read())
        visitor = Visitor()
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

    def assignLabelsToFiles(self, file1, file2, file3 = None):
        graph1, labels1 = self.assignLabels(file1)
        graph2, labels2 = self.assignLabels(file2)

        if file3 is not None:
            graph3, labels3 = self.assignLabels(file3)
            x_train = graph1 + graph2 + graph3
            y_train = labels1 + labels2 + labels3
        else:
            x_train = graph1 + graph2 
            y_train = labels1 + labels2
        
        
        return x_train, y_train

    def convertToMatrix(self, x_list):
        matrices = []
        for graph in x_list:
            matrix = nx.to_numpy_array(graph)
            # matrix = tf.convert_to_tensor(matrix, dtype=np.float32)
            matrices.append(matrix)
        return matrices

    def getDatasets(self, x_train, y_train):
        x_train_matrix = self.convertToMatrix(x_train)
        y_train = tf.keras.utils.to_categorical(y_train)  

        return x_train, x_train_matrix, y_train, 

    def getAdjacencyLists(self, x_train, x_matrix):
        embeddings = list(x_train.nodes)
        adjList = nx.dfs_successors(x_train)
        adjacencies = []
        print(end=".")
        for i in range(len(embeddings)):
            node = embeddings[i]
            embeddings[i] = sum(embeddings[i] * x_matrix[i])

            for item in adjList:
                if item == node:
                    adjacencies.append([embeddings[i], node, adjList[item]])
        return embeddings, adjacencies

    def readFiles(self, multi: bool):
        current_dir = dirname(__file__)

        merge = "./Data/Merge Sort"
        quick = "./Data/Quick Sort"

        merge = join(current_dir, merge)
        quick = join(current_dir, quick)

        if multi is True:
            other = "./Data/Other"
            other = join(current_dir, other)
            xTrain, yTrain = self.assignLabelsToFiles(merge, quick, other)
        else:
            xTrain, yTrain = self.assignLabelsToFiles(merge, quick)

        x_train_nodes, x_train_matrix, y_train = self.getDatasets(xTrain, yTrain)

        return x_train_nodes, x_train_matrix, y_train

    # def getGraphs(self, multi: bool):
    #     current_dir = dirname(__file__)

    #     merge = "./Data/Merge Sort"
    #     quick = "./Data/Quick Sort"

    #     merge = join(current_dir, merge)
    #     quick = join(current_dir, quick)

    #     if multi is True:
    #         other = "./Data/Other"
    #         other = join(current_dir, other)
    #         xTrain, yTrain = self.assignLabelsToFiles(merge, quick, other)
    #     else:
    #         xTrain, yTrain = self.assignLabelsToFiles(merge, quick)

    #     return xTrain, yTrain

    def prepareData(self, x_list, x_matrix):
        exitGraph = []
        x_nodes = x_list
        print("Collecting node embeddings and adjacency lists")
        adj = [0] * len(x_list)
        for i in range(len(x_list)):
            x_list[i], adj[i] = self.getAdjacencyLists(x_list[i], x_matrix[i])

        # for i in range(len(x_list)):
        #     x_list[i] = tf.convert_to_tensor(x_list[i], dtype=np.float32)

        index = 0
        for graph in x_list:  
            finalWorkingGraph = []
            adjacencies = adj[index]
            originalNodes = x_nodes[index]
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


class Visitor(ast.NodeVisitor):
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.adjList = []

    def generic_visit(self, node):
        if node not in self.nodes:
            nodeEmbedding = self.visitSpecial(node)
            nodeEmbedding = 1/hash(node) + 1/hash(nodeEmbedding) * 0.005
            self.nodes.append(nodeEmbedding)

        if isinstance(node, ast.AST):
            for child in list(ast.iter_child_nodes(node)):
                childEmbedding = self.visitSpecial(child)
                childEmbedding = 1/hash(child) + 1/hash(childEmbedding) * 0.005
                self.edges.append([nodeEmbedding, childEmbedding])

                if child not in self.nodes:
                    childEmbedding = self.visitSpecial(child)
                    childEmbedding = 1/hash(child) + 1/hash(childEmbedding) * 0.005
                    self.nodes.append(childEmbedding)
                    self.generic_visit(child)

        elif isinstance(node, list):
            for child in list(ast.iter_child_nodes(node)):
                childEmbedding = self.visitSpecial(child)
                childEmbedding = 1/hash(child) + 1/hash(childEmbedding) * 0.005
                self.edges.append([nodeEmbedding, childEmbedding])

                if child not in self.nodes:
                    childEmbedding = self.visitSpecial(child)
                    childEmbedding = 1/hash(child) + 1/hash(childEmbedding) * 0.005
                    self.nodes.append(childEmbedding)
                    self.generic_visit(child)

    def createAdjList(self):
        for node in self.nodes:
            children = list(ast.iter_child_nodes(node))
            if len(children) > 0:
                self.adjList.append([1/hash(node), [1/hash(child) for child in children]])

    def splitCamelCase(self, identifier: str):
        splitIdentifier = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [i.group(0) for i in splitIdentifier]

    def splitSnakeCase(self, identifier: str):
        return identifier.split("_")

    def splitIdentifier(self, identifier):
        splitId = self.splitSnakeCase(identifier)
        finalSplitID = []
        idParts = []
        for part in splitId:
            if len(part) > 0:
                idParts.append(self.splitCamelCase(part))

        if len(idParts) == 0:
            return [identifier]
        else:
            for i in idParts:
                for j in i:
                    finalSplitID.append(j)

        return finalSplitID

    def convertToGraph(self):
        graph = nx.DiGraph()
        graph.add_edges_from(self.edges)
        return graph

    def visitSpecial(self, node):
        if isinstance(node, ast.FunctionDef or ast.AsyncFunctionDef or ast.ClassDef):
            return self.visitDef(node)
        elif isinstance(node, ast.Return):
            return self.visitReturn(node)
        elif isinstance(node, ast.Delete):
            return self.visitDelete(node)
        elif isinstance(node, ast.Attribute):
            return self.visitAttribute(node)
        elif isinstance(node, ast.Assign):
            return self.visitAssign(node)
        elif isinstance(node, ast.AugAssign or ast.AnnAssign):
            return self.visitAugAssign(node)
        elif isinstance(node, ast.Attribute):
            return self.visitAttribute(node)
        elif isinstance(node, ast.Name):
            return self.visitName(node)
        elif isinstance(node, ast.Constant):
            return self.visitConstant(node)
        else:
            className = 'value = ' + node.__class__.__name__
            return className

    def visitDef(self, node: ast.FunctionDef or ast.AsyncFunctionDef or ast.ClassDef):
        return str(node.name)

    def visitReturn(self, node: ast.Return):
        returnValue = "return " + str(node.value)
        return returnValue

    def visitDelete(self, node: ast.Delete):
        returnValue = "delete " + str(node.targets)
        return returnValue

    def visitAssign(self, node: ast.Assign):
        returnValue = "assign " + str(node.value) + " to " + str(node.targets)
        return returnValue

    def visitAugAssign(self, node: ast.AugAssign or ast.AnnAssign):
        returnValue = "assign " + str(node.value) + " to " + str(node.target)
        return returnValue

    def visitAttribute(self, node: ast.Attribute):
        returnValue = str(node.attr) + " = " + str(node.value)
        return returnValue

    def visitName(self, node: ast.Name):
        return (str)

    def visitConstant(self, node: ast.Constant):
        return "value = " + str(node.value)