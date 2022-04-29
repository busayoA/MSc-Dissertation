import ast, Node
import readFiles as rf
import networkx as nx
from ete3 import Tree
# from anytree import AnyNode, RenderTree, Node

class ASTTree(ast.NodeVisitor):
    def __init__(self, startNode):
        self.root = startNode
        self.children = []
        self.nodes = []
        self.nodeIDs = []
        self.edgeSets = []

    def insertRelationship(self, parentNode, childNode):
        if parentNode.nodeID not in childNode.childIDs:
            if parentNode.nodeID not in self.nodeIDs:
                self.nodes.append(parentNode)
                self.nodeIDs.append(parentNode.nodeID)

            if childNode.nodeID not in self.nodeIDs:
                self.nodes.append(childNode)
                self.nodeIDs.append(childNode.nodeID)

            if childNode.nodeID not in parentNode.childIDs:
                parentNode.addChild(childNode)
                self.edgeSets.append([parentNode.nodeObject, childNode.nodeObject])

    def traverseAST(self, startNode):
        parentNode = Node.Node(startNode)
        if isinstance(startNode, ast.AST):
            for node in list(ast.iter_child_nodes(startNode)):
                childNode = Node.Node(node)
                self.traverseAST(node)
                self.insertRelationship(parentNode, childNode)
        elif isinstance(startNode, list):
            for node in list(ast.iter_child_nodes(startNode)):
                childNode = Node.Node(node)
                self.traverseAST(node)
                self.insertRelationship(parentNode, childNode)
        
        print(parentNode.nodeToString(), ":", [childNode.nodeToString() for child in parentNode.children])
        # elif isinstance(startNode, list):
        #     for node in list(ast.iter_child_nodes(startNode)):
        #         print(node)
        # # return list(self.children)

    def visualiseTree(self):
        G = nx.DiGraph()
        G.add_edges_from(self.edgeSets)
        subtrees = {node:Tree(name=node) for node in G.nodes()}
        [*map(lambda edge:subtrees[edge[0]].add_child(subtrees[edge[1]]), G.edges())]

        t = subtrees[self.root]
        # print(t.get_ascii())
        t.show()
        # plt.show()
        # plt.savefig("filename.png")
        return t

merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort/1.py"
def assignLabels():
    with open (merge, "r") as file:
        return ast.parse(file.read())

f = assignLabels()
astTree = ASTTree(f)
astTree.traverseAST(f)
astTree.visualiseTree()
