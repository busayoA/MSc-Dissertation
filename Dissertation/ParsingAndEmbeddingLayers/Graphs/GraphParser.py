import os, ast, random, sys
import networkx as nx
from ParsingAndEmbeddingLayers.Visitor import Visitor, HashVisitor
from os.path import dirname, join

class GraphParser:
    def __init__(self, hashed: bool):
        """
        Initliaise the GraphParser object with a truth value for hashed
        hashed: bool - Whether or not we are to work with hashed data
        """
        self.hashed = hashed

    def convertToGraph(self, filePath):
        """
        Read a file and convert its contents into its graph representation

        filePath - The file that is to be read

        Returns:
        graph: The graph representation of the contents of the file
        """
        programAST = ''
        with open (filePath, "r") as file:
            programAST = ast.parse(file.read())

        # determine if the values are to be hashed or unhashed
        if self.hashed is True:
            visitor = HashVisitor()
            visitor.generic_visit(programAST)
        else:
            visitor = Visitor()
            visitor.generic_visit(programAST)
        graph = visitor.convertToGraph()
        return graph

    def assignLabels(self, filePath):
        """
        Assign class labels to each file and its corresponding graph based on 
        the sorting algorithm it implements. 0 for Merge Sort and 1 for Quick Sort.

        filePath - The path to the set of files

        Returns:
        graphs - The graph representations of the file contents
        labels - The class labels corresponding to the graphs
        """
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
        return graphs, labels

    def assignLabelsToFiles(self, file1, file2):
        """
        Call the assignLabels method on both the merge and quick sort files

        file1: The file path of the merge sort
        file2: The file path of the quick sort

        Returns:
        x - The graphs from all the sorting algorithms
        y - The class labels from all the files 
        """
        graph1, labels1 = self.assignLabels(file1)
        graph2, labels2 = self.assignLabels(file2)

        x = graph1 + graph2 
        y = labels1 + labels2
        
        return x, y

    def convertToMatrix(self, x):
        """
        Convert an individual graph to its matrix representation using the NetworkX API

        x: The graph to be converted into a matrix

        Returns:
        matrices: The matrix representation of the graph in x
        """
        matrices = []
        for graph in x:
            matrix = nx.to_numpy_array(graph)
            matrices.append(matrix)
        return matrices

    def extractNodes(self, graphs):
        """
        Given a set of NetworkX graphs, extract the nodes from each graph

        graphs - The graphs from which nodes are to be extracted

        Returns:
        xNodes - The nodes extracted from the graphs in graphs
        """
        xNodes = []
        for i in range(len(graphs)):
            currentGraph = list(graphs[i].nodes)
            xNodes.append(currentGraph)
        return xNodes

    def readFiles(self):
        """
        Tie all the above methods together and parse the files 

        Returns;
        x - The list of graphs and nodes in pairs of tuples
        x_graph - The lsit of graphs on its own
        matrices - THe matrix representations of the graphs in x_graphs
        labels - The class labels
        """
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


