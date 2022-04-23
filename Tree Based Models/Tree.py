import ast, Node, astpretty
from tracemalloc import start
import readFiles as rf
import networkx as nx
from ete3 import Tree
import matplotlib.pyplot as plt
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
        root = "body"
        subtrees = {node:Tree(name=node) for node in G.nodes()}
        [*map(lambda edge:subtrees[edge[0]].add_child(subtrees[edge[1]]), G.edges())]

        t = subtrees[self.root]
        print(t.get_ascii())
        t.show()
        # plt.show()
        # plt.savefig("filename.png")

merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort/1.py"
def assignLabels():
    with open (merge, "r") as file:
        return ast.parse(file.read())

f = assignLabels()
astTree = ASTTree(f)
astTree.traverseAST(f)
astTree.visualiseTree()

# class Tree1:
#     def __init__(self):
#         self.nodes = []
#         self.nodeIDs = []
#         self.edgeSets = []

#     def insertRelationship(self, parentNode, childNode):
#         if parentNode.nodeID not in self.nodeIDs:
#             self.nodes.append(parentNode)
#             self.nodeIDs.append(parentNode.nodeID)

#         if childNode.nodeID not in self.nodeIDs:
#             self.nodes.append(childNode)
#             self.nodeIDs.append(childNode.nodeID)

#         if parentNode.nodeID not in childNode.childIDs:
#             if childNode.nodeID not in parentNode.childIDs:
#                 parentNode.addChild(childNode)
#                 self.edgeSets.append([parentNode.nodeID, childNode.nodeID])

#     def printAdjList(self):
#         for node in self.nodes:
#             node.printNode()

#     def visualiseTree(self):
#         G = nx.DiGraph()
#         G.add_edges_from(self.edgeSets)
#         root = "A"
#         subtrees = {node:Tree(name=node) for node in G.nodes()}
#         [*map(lambda edge:subtrees[edge[0]].add_child(subtrees[edge[1]]), G.edges())]

#         t = subtrees[root]
#         print(t.get_ascii())
#         # plt.show()
#         # plt.savefig("filename.png")


# nodeA = Node.Node({'id':'A'})
# nodeB = Node.Node({'id':'B'})
# nodeC = Node.Node({'id':'C'})
# nodeD = Node.Node({'id':'D'})
# nodeE = Node.Node({'id':'E'})
# nodeF = Node.Node({'id':'F'})

# myTree = Tree1()
# myTree.insertRelationship(nodeA, nodeB)
# myTree.insertRelationship(nodeA, nodeC)
# myTree.insertRelationship(nodeA, nodeD)
# myTree.insertRelationship(nodeB, nodeE)
# myTree.insertRelationship(nodeB, nodeF)
# myTree.insertRelationship(nodeB, nodeA)

# # myTree.visualiseTree()

# merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort/1.py"
# def assignLabels():
#     with open (merge, "r") as file:
#         return ast.parse(file.read())

# f = assignLabels()
# function1 = f.body[0]
# # print(function1.b)
# # astpretty.pprint(f)
# print(getattr(f, str(f._fields[0])))

# def generateTree(startNode):
#     if isinstance(startNode, ast.AST):
#         node = [generateTree(getattr(startNode, k)) for k in startNode._fields]
#         return node
#     elif isinstance(startNode, list):
#         return [generateTree(el) for el in startNode]
#     else:
#         return startNode

# print(generateTree(f))
#     tree, label, sub_tokens, size, file_path = tree_data["tree"], tree_data["label"], tree_data["sub_tokens"] , tree_data["size"], tree_data["file_path"]
#     print("Extracting............", file_path)
#     # print(tree)
#     node_type_id = []
#     node_token = []
#     node_sub_tokens_id = []
#     node_index = []

#     children_index = []
#     children_node_type_id = []
#     children_node_token = []
#     children_node_sub_tokens_id = []
#     # label = 0

#     # print("Label : " + str(label))
#     queue = [(tree, -1)]
#     # print queue
#     while queue:
#         # print "############"
#         node, parent_ind = queue.pop(0)
#         # print node
#         # print parent_ind
#         node_ind = len(node_type_id)
#         # print "node ind : " + str(node_ind)
#         # add children and the parent index to the queue
#         queue.extend([(child, node_ind) for child in node['children']])
#         # create a list to store this node's children indices
#         children_index.append([])
#         children_node_type_id.append([])
#         children_node_token.append([])
#         children_node_sub_tokens_id.append([])
#         # add this child to its parent's child list
#         if parent_ind > -1:
#             children_index[parent_ind].append(node_ind)
#             children_node_type_id[parent_ind].append(int(node["node_type_id"]))
#             children_node_token[parent_ind].append(node["node_token"])
#             children_node_sub_tokens_id[parent_ind].append(node["node_sub_tokens_id"])
#         # print("a")
#         # print(children_node_types)
#         # print("b")
#         # print(children_node_sub_tokens_id)
#         node_type_id.append(node['node_type_id'])
#         node_token.append(node['node_token'])
#         node_sub_tokens_id.append(node['node_sub_tokens_id'])
#         node_index.append(node_ind)

#     results = {}
#     results["node_index"] = node_index
#     results["node_type_id"] = node_type_id
#     results["node_token"] = node_token
#     results["node_sub_tokens_id"] = node_sub_tokens_id
#     results["children_index"] = children_index
#     results["children_node_type_id"] = children_node_type_id
#     results["children_node_token"] = children_node_token
#     results["children_node_sub_tokens_id"] = children_node_sub_tokens_id
#     results["size"] = size
#     results["label"] = label

#     return results
    # if isinstance(startNode, ast.AST):
    #     functions = startNode._fields
    #     for func in functions:
    #         tree = generateTree(func)
    #     return tree
    # elif isinstance(startNode, list):
    #     tree = [generateTree(node) for node in startNode]
    #     return tree
    # else:
    #     tree = startNode
    #     return tree

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