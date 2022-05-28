import ast
import re
from cv2 import split
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

from ete3 import Tree

mpl.use('Qt5Agg')

class Node(ast.NodeVisitor):
    def __init__(self):
        self.nodeList = []
        self.edgeSets = []
        self.adjList = []

    def visit(self, node):
        method = '' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        if 1/hash(node) not in self.nodeList:
            self.nodeList.append(1/hash(node))
        if isinstance(node, ast.AST):
            for child in list(ast.iter_child_nodes(node)):
                if 1/hash(child) not in self.nodeList:
                    self.nodeList.append(1/hash(child))
                    self.edgeSets.append([1/hash(node), 1/hash(child)])
                    self.generic_visit(child)
        elif isinstance(node, list):
            for child in list(ast.iter_child_nodes(node)):
                if 1/hash(child) not in self.nodeList:
                    self.nodeList.append(1/hash(child))
                    self.edgeSets.append([1/hash(node), 1/hash(child)])
                    self.generic_visit(child)
            
    def getChildren(self, node):
        children = list(ast.iter_child_nodes(node))
        return children

    def createAdjList(self):
        for node in self.nodeList:
            nodeRelationships = []
            for i in self.edgeSets:
                if i[0] == node:
                    nodeRelationships.append(1/hash(i[1]))
            if len(nodeRelationships) == 0:
                self.adjList.append([1/hash(node)])    
            else:
                self.adjList.append([1/hash(node), nodeRelationships])

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

        if len(idParts) is 0:
            return [identifier]
        else:
            for i in idParts:
                for j in i:
                    finalSplitID.append(j)

        return finalSplitID


merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort/1.py"
def readAST():
    with open (merge, "r") as file:
        return ast.parse(file.read())

programAST = readAST()

# CREATE THE AST GRAPH
node = Node()
node.generic_visit(programAST)


# print(node.nodeList)
node.createAdjList()

[print(node.adjList[i]) for i in range(len(node.adjList))]
# print(node.nodeList)

# G = nx.DiGraph()
# G.add_edges_from(node.edgeSets)
# nx.draw_networkx(G)
# # plt.show()

# g = tf.Graph()

# # CREATE A TREE FROM THE AST GRAPH
# subtrees = {node:Tree(name=node) for node in G.nodes()}
# [*map(lambda edge:subtrees[edge[0]].add_child(subtrees[edge[1]]), G.edges())]

# t = subtrees[1/hash(programAST)]
# # print(t.get_ascii())
# t.show()




