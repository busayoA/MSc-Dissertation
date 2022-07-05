import os, ast, re, random
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
            x = graph1 + graph2 + graph3
            y = labels1 + labels2 + labels3
        else:
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

    def readFiles(self, multi: bool):
        current_dir = dirname(__file__)

        merge = "./Data/Merge Sort"
        quick = "./Data/Quick Sort"

        merge = join(current_dir, merge)
        quick = join(current_dir, quick)

        if multi is True:
            other = "./Data/Other"
            other = join(current_dir, other)
            x_graph, y = self.assignLabelsToFiles(merge, quick, other)
        else:
            x_graph, y  = self.assignLabelsToFiles(merge, quick)

        matrices = self.convertToMatrix(x_graph)
        x_list, matrices = self.padGraphs(x_graph, matrices)
        x = []
        for i in range(len(x_list)):
            x.append([x_list[i], x_graph[i]])

        for i in range(len(x)):
            x[0][0].append(y[i])
        
        random.shuffle(x)
        labels = []
        for i in range(len(x)):
            labels.append(x[i][0][-1])
            x[i][0].pop()

        return x, matrices, labels

    def padGraphs(self, graphs, matrices):
        x = []
        maxLen = 0
        for i in graphs:
            if maxLen < len(i):
                maxLen = len(i)

        for i in range(len(graphs)):
            currentGraph = list(graphs[i].nodes)
            currentMatrix = matrices[i]

            if len(currentGraph) < maxLen:
                shape = np.shape(currentMatrix)
                padCount = maxLen - len(currentGraph)

                paddedMatrix = np.zeros((maxLen, maxLen))
                paddedMatrix[:shape[0], :shape[1]] = currentMatrix

                for j in range(padCount):
                    currentGraph.append(0.0)
            x.append(currentGraph)
            matrices[i] = paddedMatrix

        return x, matrices

    def getAdjacencyLists(self, x, matrices):
        embeddings = list(x.nodes)
        adjList = nx.dfs_successors(x)
        adjacencies = []
        print(end=".")
        for i in range(len(embeddings)):
            node = embeddings[i]
            embeddings[i] = sum(embeddings[i] * matrices[i])

            for item in adjList:
                if item == node:
                    adjacencies.append([embeddings[i], node, adjList[item]])
        return embeddings, adjacencies

    def prepareData(self, x, matrices):
        exitGraph = []
        nodes = x
        print("Collecting node embeddings and adjacency lists")
        adj = [0] * len(x)
        for i in range(len(x)):
            x[i], adj[i] = self.getAdjacencyLists(x[i], matrices[i])

        # for i in range(len(x_list)):
        #     x_list[i] = tf.convert_to_tensor(x_list[i], dtype=np.float32)

        index = 0
        for graph in x:  
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

    def splitTrainTest(self, x, matrices, y):
        split = int(0.7 * len(x))

        x_train = x[:split]
        x_train_matrix = matrices[:split]
        y_train = y[:split]

        x_test = x[split:]
        x_test_matrix = matrices[split:]
        y_test = y[split:]

        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)  
        
        return x_train, x_train_matrix, y_train, x_test, x_test_matrix, y_test


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
                    self.nodes.append(childEmbedding)
                    self.generic_visit(child)

        elif isinstance(node, list):
            for child in list(ast.iter_child_nodes(node)):
                childEmbedding = self.visitSpecial(child)
                childEmbedding = 1/hash(child) + 1/hash(childEmbedding) * 0.005
                self.edges.append([nodeEmbedding, childEmbedding])

                if child not in self.nodes:
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
