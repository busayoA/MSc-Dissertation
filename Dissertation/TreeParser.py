import os, ast
from TreeNode import createTreeFromEdges
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
            tree = createTreeFromEdges(visitor.edges)
        else:
            visitor = Visitor()
            visitor.generic_visit(programAST)
            tree = createTreeFromEdges(visitor.edges)
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
