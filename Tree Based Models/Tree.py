import ast, Edge, Node, astpretty
import readFiles as rf
import networkx as nx
from ete3 import Tree
import matplotlib.pyplot as plt
# from anytree import AnyNode, RenderTree, Node

class Tree1:
    def __init__(self):
        self.nodes = []
        self.nodeIDs = []
        self.edgeSets = []

    def insertRelationship(self, parentNode, childNode):
        if parentNode.nodeID not in self.nodeIDs:
            self.nodes.append(parentNode)
            self.nodeIDs.append(parentNode.nodeID)

        if childNode.nodeID not in self.nodeIDs:
            self.nodes.append(childNode)
            self.nodeIDs.append(childNode.nodeID)

        if parentNode.nodeID not in childNode.childIDs:
            if childNode.nodeID not in parentNode.childIDs:
                parentNode.addChild(childNode)
                self.edgeSets.append([parentNode.nodeID, childNode.nodeID])

    def printAdjList(self):
        for node in self.nodes:
            node.printNode()

    def visualiseTree(self):
        G = nx.DiGraph()
        G.add_edges_from(self.edgeSets)
        
        root = "A"
        subtrees = {node:Tree(name=node) for node in G.nodes()}
        [*map(lambda edge:subtrees[edge[0]].add_child(subtrees[edge[1]]), G.edges())]

        t = subtrees[root]
        print(t.get_ascii())
        # plt.show()
        # plt.savefig("filename.png")


nodeA = Node.Node({'id':'A'})
nodeB = Node.Node({'id':'B'})
nodeC = Node.Node({'id':'C'})
nodeD = Node.Node({'id':'D'})
nodeE = Node.Node({'id':'E'})
nodeF = Node.Node({'id':'F'})

myTree = Tree1()
myTree.insertRelationship(nodeA, nodeB)
myTree.insertRelationship(nodeA, nodeC)
myTree.insertRelationship(nodeA, nodeD)
myTree.insertRelationship(nodeB, nodeE)
myTree.insertRelationship(nodeB, nodeF)
myTree.insertRelationship(nodeB, nodeA)

# myTree.visualiseTree()

merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort/1.py"
def assignLabels():
    with open (merge, "r") as file:
        return ast.parse(file.read())

f = assignLabels()
astpretty.pprint(f)

# t = Tree(str(f), format=8)

# print(t)

# class TreeNode(ast.NodeVisitor, Node):
#     def __init__(self, ID, contents):
#         self.contents = contents
#         self.childrenNodes = []
#         self.node = AnyNode(id="root")

#     def printNode(self):
#         print("\nData:", self.childrenNodes)

#     def visitFunction(self, node):
#         self.visit(node)
#         self.generic_visit(node)

#     # def generateTree(self, startNode):
#     #     if isinstance(startNode, ast.AST):
#     #         functions = startNode._fields
#     #         for func in functions:
#     #             tree = self.generateTree(func)
#     #         return tree
#     #     elif isinstance(startNode, list):
#     #         tree = [self.generateTree(node) for node in startNode]
#     #         return tree
#     #     else:
#     #         tree = startNode
#     #         return tree

#     def generateTree(self, startNode):
#         if isinstance(startNode, ast.FunctionDef):
#             AnyNode(id="root")

#     # def visitNode(self, node):
#     #     if isinstance(node, ast.FunctionDef):
#     #         AnyNode(id=node.name, parent=self.node)
#     #         print("Body", node.body)
#     #         childNodes = node.body
#     #         for child in childNodes:
#     #             self.visitNode(child)
#     #     # elif isinstance(node, ast.Assign):
#     #     #     self.visit(node)
#     #     #     print("Assignment", node.targets)
#     #     # elif isinstance(node, ast.Return):
#     #     #     pass
#     #     # elif isinstance(node, ast.For):
#     #     #     pass
#     #     elif isinstance(node, ast.stmt):
#     #         pass
#     #     else:
#     #         print("Node", node)


# treeList = rf.assignLabels()
# rootNode = TreeNode(treeList[0], {'name':'module', 'type':'Python Module'})
# print(rootNode.generateTree(treeList[0].body))
# # for i in range(len(treeList[0].body)):
# #     rootNode.visitNode(treeList[0].body[i])
# # print(treeList[0])
# # print(RenderTree(rootNode.node))
# # functions = treeList[0].body
# # for func in functions:
# #     rootNode.visitNode(func)


# # rootNode.printNode()

# # trees = rf.getPythonAST()
# # # print(trees[0].body)
# # # FuncLister().visit(trees[1])
# # # print(ast.walk())
# # tree1 = trees[0]
# # tree1A = ast.dump(tree1)
# # treeList = tree1A.split("[")
# # # [print(a, end = "\n") for a in treeList]
# # # for node in ast.walk(trees[0]):
# # #     if isinstance(node, ast.BinOp):
# # #         print(node.left)
# # rootNodes = []

# # for node in ast.walk(tree1):
# #     if isinstance(node, ast.FunctionDef):
# #         rootNodes.append(node.name)

# # print(rootNodes)