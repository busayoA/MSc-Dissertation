import ast
from pydoc import classname
import re
from cv2 import split
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

from ete3 import Tree

mpl.use('Qt5Agg')

class Node(ast.NodeVisitor):
    def __init__(self):
        self.nodes = []
        self.astFields = []

        self.edgeSets = []
        self.edges = []

        self.adjList = []
        self.hashedAdjList = []

    def visit(self, node):
        method = '' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def getClassName(self, node):
        className = "" + node.__class__.__name__
        return className

    def generic_visit(self, node):
        if node not in self.nodes:
            self.nodes.append(node)

        if isinstance(node, ast.AST):
            for child in list(ast.iter_child_nodes(node)):
                if child not in self.nodes:
                    self.nodes.append(child)
                    self.edges.append([node, child])
                    self.generic_visit(child)

        elif isinstance(node, list):
            for child in list(ast.iter_child_nodes(node)):
                if child not in self.nodes:
                    self.nodes.append(child)
                    self.edges.append([node, child])
                    self.generic_visit(child)

    def getChildren(self, node):
        children = list(ast.iter_child_nodes(node))
        return children

    def createAdjList(self):
        for node in self.nodes:
            children = list(ast.iter_child_nodes(node))
            for child in children:
                self.edges.append([node, child])
            self.adjList.append([node, children])
            if len(children) > 1:
                for i in range(len(children)):
                    j = i+1
                    currentChild = children[i]
                    if len(children[j:]) > 0:
                        for k in range(len(children[j:])):
                            self.edges.append([currentChild, child])
                        self.adjList.append([currentChild, children[j:]])


            

        # for i in range(len(self.nodeHashes)):
        #     currentNode = self.nodeList[i]
        #     currentNodeHash = self.nodeHashes[i]

        #     nodeRelationships = []
        #     hashedNodeRelationships = []
        #     for j in range(len(self.edgeSets)):
        #         currentSet = self.edgeSets[j]
        #         currentHash = self.edgeHashes[j]

        #         if currentHash[0] == currentNodeHash:
        #             nodeRelationships.append(currentSet[1])
        #             hashedNodeRelationships.append(currentHash[1])

        #     if len(nodeRelationships) == 0:
        #         self.adjList.append([currentNode])   
        #         self.hashedAdjList.append([currentNodeHash])    
        #     else:
        #         self.adjList.append([currentNode, nodeRelationships])
        #         self.hashedAdjList.append([currentNodeHash, hashedNodeRelationships])

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
# print(ast.dump(programAST))

# print(node.nodes)
node.createAdjList()

[print(node.adjList[i]) for i in range(len(node.adjList))]
# # print(node.nodeList)

G = nx.DiGraph()
G.add_edges_from(node.edgeSets)
nx.draw_networkx(G)
plt.show()

# g = tf.Graph()

# # # CREATE A TREE FROM THE AST GRAPH
# subtrees = {node:Tree(name=node) for node in G.nodes()}
# [*map(lambda edge:subtrees[edge[0]].add_child(subtrees[edge[1]]), G.edges())]

# t = subtrees[1/hash(programAST)]
# # print(t.get_ascii())
# t.show()




