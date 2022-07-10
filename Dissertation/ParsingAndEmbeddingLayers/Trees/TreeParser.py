import os, ast
from TreeNode import TreeNode
from Visitor import Visitor, HashVisitor

class TreeParser():
    def __init__(self, hashed: bool):
        self.hashed = hashed

    def convertToTree(self, filePath):
        programAST = ''
        with open (filePath, "r") as file:
            programAST = ast.parse(file.read())

        if self.hashed is True:
            visitor = HashVisitor()
            visitor.generic_visit(programAST)
            tree = self.createTreeFromEdges(visitor.edges)
        else:
            visitor = Visitor()
            visitor.generic_visit(programAST)
            tree = self.createTreeFromEdges(visitor.edges)
        return tree

    def parse(self, filePath):
        trees, labels = [], []
        os.chdir(filePath)
        for file in os.listdir():
            # Check whether file is in text format or not
            if file.endswith(".py"):
                path = f"{filePath}/{file}"
                # call read text file function
                tree = self.convertToTree(path)
                trees.append(tree)
                if filePath.find("Merge") != -1:
                    labels.append(0)
                elif filePath.find("Quick") != -1:
                    labels.append(1)
        return trees, labels

    def createTreeFromEdges(self, edges):
        individualNodes = []
        for edge in edges:
            for i in edge:
                if i not in individualNodes:
                    individualNodes.append(i)
        nodes = {i: TreeNode(i) for i in individualNodes}
        for parent, child in edges:
            if child not in nodes[parent].children:
                nodes[parent].children.append(nodes[child])
            if child in individualNodes:
                individualNodes.remove(child)
        for edge in individualNodes:
            return nodes[edge]
