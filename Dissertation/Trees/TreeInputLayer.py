import os, ast
import tensorflow as tf
from os.path import dirname, join

def createTreeFromEdges(edges):
    nodeEdges = set(i for edge in edges for i in edge)
    nodes = {embedding: Node(embedding) for embedding in nodeEdges}

    for parent, child in edges:
        if child not in nodes[parent].children:
            nodes[parent].children.append(nodes[child])
        if child in nodeEdges:
            nodeEdges.remove(child)
    
    for edge in nodeEdges:
        return nodes[edge]

class TreeInputLayer():
    def convertToTree(self, filePath):
        programAST = ''
        with open (filePath, "r") as file:
            programAST = ast.parse(file.read())
        visitor = Visitor()
        visitor.generic_visit(programAST)
        tree = createTreeFromEdges(visitor.edges)

        return tree

    def assignLabels(self, filePath):
        current_dir = dirname(__file__)
        filePath = join(current_dir, filePath)
        trees, labels = [], []
        os.chdir(filePath)
        for file in os.listdir():
            # Check whether file is in text format or not
            if file.endswith(".py"):
                path = f"{filePath}/{file}"
                # call read text file function
                treeData = self.convertToTree(path)
                trees.append(treeData)
                if filePath.find("Merge") != -1:
                    labels.append(0)
                elif filePath.find("Quick") != -1:
                    labels.append(1)
                elif filePath.find("Other") != -1:
                    labels.append(2)
        return trees, labels

    def getData(self, multi: bool):
        current_dir = dirname(__file__)

        merge = "./Data/Merge Sort"
        quick = "./Data/Quick Sort"
        other = "./Data/Other"

        merge = join(current_dir, merge)
        quick = join(current_dir, quick)

        mergeTree, mergeLabels = self.assignLabels(merge)
        quickTree, quickLabels = self.assignLabels(quick)
        if multi is True:
            otherTree, otherLabels = self.assignLabels(other)

            x_train = mergeTree + quickTree + otherTree
            y_train = mergeLabels + quickLabels + otherLabels
        else:
            x_train = mergeTree + quickTree 
            y_train = mergeLabels + quickLabels 
        y_train = tf.keras.utils.to_categorical(y_train)  

        return x_train, y_train


class Visitor(ast.NodeVisitor):
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.adjList = []

    def generic_visit(self, node):
        if node not in self.nodes:
            nodeEmbedding = self.visitSpecial(node)
            nodeEmbedding = 1/hash(node) + 1/hash(nodeEmbedding) * 0.005
            self.nodes.append(nodeEmbedding)

        if isinstance(node, ast.AST):
            for child in list(ast.iter_child_nodes(node)):
                childEmbedding = self.visitSpecial(child)
                childEmbedding = 1/hash(child) + 1/hash(childEmbedding) * 0.005
                self.edges.append([nodeEmbedding, childEmbedding])

                if child not in self.nodes:
                    childEmbedding = self.visitSpecial(child)
                    childEmbedding = 1/hash(child) + 1/hash(childEmbedding) * 0.005
                    self.nodes.append(childEmbedding)
                    self.generic_visit(child)

        elif isinstance(node, list):
            for child in list(ast.iter_child_nodes(node)):
                childEmbedding = self.visitSpecial(child)
                childEmbedding = 1/hash(child) + 1/hash(childEmbedding) * 0.005
                self.edges.append([nodeEmbedding, childEmbedding])

                if child not in self.nodes:
                    childEmbedding = self.visitSpecial(child)
                    childEmbedding = 1/hash(child) + 1/hash(childEmbedding) * 0.005
                    self.nodes.append(childEmbedding)
                    self.generic_visit(child)

    def visitSpecial(self, node):
        if isinstance(node, ast.FunctionDef or ast.AsyncFunctionDef or ast.ClassDef):
            return self.visitDef(node)
        elif isinstance(node, ast.Return):
            return self.visitReturn(node)
        elif isinstance(node, ast.Delete):
            return self.visitDelete(node)
        elif isinstance(node, ast.Attribute):
            return self.visitAttribute(node)
        elif isinstance(node, ast.Assign):
            return self.visitAssign(node)
        elif isinstance(node, ast.AugAssign or ast.AnnAssign):
            return self.visitAugAssign(node)
        elif isinstance(node, ast.Attribute):
            return self.visitAttribute(node)
        elif isinstance(node, ast.Name):
            return self.visitName(node)
        elif isinstance(node, ast.Constant):
            return self.visitConstant(node)
        else:
            className = 'value = ' + node.__class__.__name__
            return className

    def visitDef(self, node: ast.FunctionDef or ast.AsyncFunctionDef or ast.ClassDef):
        return str(node.name)

    def visitReturn(self, node: ast.Return):
        returnValue = "return " + str(node.value)
        return returnValue

    def visitDelete(self, node: ast.Delete):
        returnValue = "delete " + str(node.targets)
        return returnValue

    def visitAssign(self, node: ast.Assign):
        returnValue = "assign " + str(node.value) + " to " + str(node.targets)
        return returnValue

    def visitAugAssign(self, node: ast.AugAssign or ast.AnnAssign):
        returnValue = "assign " + str(node.value) + " to " + str(node.target)
        return returnValue

    def visitAttribute(self, node: ast.Attribute):
        returnValue = str(node.attr) + " = " + str(node.value)
        return returnValue

    def visitName(self, node: ast.Name):
        return (str)

    def visitConstant(self, node: ast.Constant):
        return "value = " + str(node.value)


class Node():
    def __init__(self, embedding: float):
        self.embedding = embedding
        self.children = []

    def preorderTraversal(self, root):
        embeddingTree = []
        objectTree = []

        if root is not None:
            embeddingTree.append(root.embedding)
            objectTree.append(root)
            for i in range(len(root.children)):
                embeddingTree = embeddingTree + (self.preorderTraversal(root.children[i]))[0]
                objectTree = objectTree + (self.preorderTraversal(root.children[i]))[1]

        return list(set(embeddingTree)), list(set(objectTree))
