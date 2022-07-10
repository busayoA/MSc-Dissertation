import os, ast, random, sys
import networkx as nx
from ParsingAndEmbeddingLayers.Visitor import Visitor, HashVisitor
from os.path import dirname, join

class GraphParser:
    def __init__(self, hashed):
        self.hashed = hashed

    def convertToGraph(self, filePath):
        programAST = ''
        with open (filePath, "r") as file:
            programAST = ast.parse(file.read())
        if self.hashed is True:
            visitor = HashVisitor()
            visitor.generic_visit(programAST)
        else:
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

    def extractNodes(self, graphs):
        xNodes = []
        for i in range(len(graphs)):
            currentGraph = list(graphs[i].nodes)
            xNodes.append(currentGraph)
        return xNodes

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

        return x, x_graph, matrices, labels


