import os, ast, dgl
import numpy as np
import tensorflow as tf
import networkx as nx
from os.path import dirname, join

class BinaryGraphInputLayer():
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

    def splitTrainTest(self, file1, file2):
        graph1, labels1 = self.assignLabels(file1)
        graph2, labels2 = self.assignLabels(file2)
        split1 = int(0.6 * len(graph1))
        split2 = int(0.6 * len(graph2))

        x_train = graph1[:split1] + graph2[:split2]
        y_train = labels1[:split1] + labels2[:split2]
        
        x_test = graph1[split1:] + graph2[split2:]
        y_test = labels1[split1:] + labels2[split2:]
        
        return x_train, y_train, x_test, y_test

    def convertToDGL(self, x_list):
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

    # def getDatasets(self, x_train, y_train, x_test, y_test):
    #     x_train_matrix = self.convertToMatrix(x_train)
    #     y_train = tf.keras.utils.to_categorical(y_train)  

    #     x_test_matrix = self.convertToMatrix(x_test)
    #     y_test = tf.keras.utils.to_categorical(y_test)  

    #     return x_train, x_train_matrix, y_train, x_test, x_test_matrix, y_test

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

    def readFiles(self):
        current_dir = dirname(__file__)

        merge = "./Data/Merge Sort"
        quick = "./Data/Quick Sort"

        merge = join(current_dir, merge)
        quick = join(current_dir, quick)

        xTrain, yTrain, xTest, yTest = self.splitTrainTest(merge, quick)
        x_train_nodes, x_train_matrix, y_train, x_test_nodes, x_test_matrix, y_test = self.getDatasets(xTrain, yTrain, xTest, yTest)

        return x_train_nodes, x_train_matrix, y_train, x_test_nodes, x_test_matrix, y_test

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