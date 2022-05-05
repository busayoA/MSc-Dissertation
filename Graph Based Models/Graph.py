import ast
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from ete3 import Tree

mpl.use('Qt5Agg')

merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort/1.py"
def readAST():
    with open (merge, "r") as file:
        return ast.parse(file.read())

programAST = readAST()


class Node(ast.NodeVisitor):
    def __init__(self):
        self.nodeList = []
        self.edgeSets = []

    def visitAllNodes(self, node):
        self.nodeList.append(node)
        if isinstance(node, ast.AST):
            for child in list(ast.iter_child_nodes(node)):
                self.nodeList.append(child)
                self.generic_visit(child)
        elif isinstance(node, list):
            for child in list(ast.iter_child_nodes(node)):
                self.nodeList.append(child)
                self.generic_visit(child)
        else:
            self.generic_visit(node)
            self.nodeList.append(node)

    def getGraph(self, root):
        self.visitAllNodes(root)
        nodes = self.nodeList
        i = 0
        while i < len(nodes):
            currentNode = self.nodeList[i]
            nodeChildren = self.getChildren(currentNode)
            for node in nodeChildren:
                self.edgeSets.append([currentNode, node])
                for miniChild in self.getChildren(node):
                    self.edgeSets.append([node, miniChild])
                    for child2 in self.getChildren(miniChild):
                        self.edgeSets.append([miniChild, child2])
            i+=1
            
    def getChildren(self, node):
        children = list(ast.iter_child_nodes(node))
        return children

class Graph():
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.edgeSets = []

    def addNode(self, node, children):
        if node not in self.nodes:
            self.nodes.append(node)
            for child in children:
                self.nodes.append(child)
                self.edgeSets.append([node, child])
    
    def visualiseGraph(self):
        G = nx.DiGraph()
        G.add_edges_from(self.edgeSets)
        nx.draw_networkx(G)
        plt.show()


node = Node()
node.getGraph(programAST)
print(node.nodeList)
print(node.edgeSets)
G = nx.DiGraph()
G.add_edges_from(node.edgeSets)
# nx.draw_networkx(G)
# plt.show()

subtrees = {node:Tree(name=node) for node in G.nodes()}
[*map(lambda edge:subtrees[edge[0]].add_child(subtrees[edge[1]]), G.edges())]

t = subtrees[programAST]
# print(t.get_ascii())
t.show()


# for child in node.getChildren(programAST):
#     children = node.getChildren(child)
#     graph.addNode(child, children)
#     for miniChild in children:
#         miniChildren = node.getChildren(miniChild)
#         graph.addNode(miniChild, miniChildren)
# graph.visualiseGraph()