import ast, re
import networkx as nx
from abc import ABC, abstractmethod

class AbstractVisitor(ast.NodeVisitor, ABC):
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.adjList = []
        self.hashedNodes = []

    @abstractmethod
    def generic_visit(self, node):
        raise NotImplementedError()

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

        if len(idParts) == 0:
            return [identifier]
        else:
            for i in idParts:
                for j in i:
                    finalSplitID.append(j)

        return finalSplitID
        
    def convertToGraph(self):
        graph = nx.DiGraph()
        graph.add_edges_from(self.edges)
        return graph

    def createAdjList(self):
        for node in self.nodes:
            children = list(ast.iter_child_nodes(node))
            if len(children) > 0:
                self.adjList.append([1/hash(node), [1/hash(child) for child in children]])

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

class Visitor(AbstractVisitor):
    def __init__(self):
        super().__init__()

    def generic_visit(self, node):
        if node not in self.nodes:
            self.nodes.append(node)

        if isinstance(node, ast.AST):
            for child in list(ast.iter_child_nodes(node)):
                for child in list(ast.iter_child_nodes(node)):
                    self.edges.append([node, child])
                    if child not in self.nodes:
                        self.nodes.append(child)
                        self.generic_visit(child)

        elif isinstance(node, list):
            for child in list(ast.iter_child_nodes(node)):
                self.edges.append([node, child])
                if child not in self.nodes:
                    self.nodes.append(child)
                    self.generic_visit(child)

class HashVisitor(AbstractVisitor):
    def __init__(self):
        super().__init__()

    def generic_visit(self, node):
        nodeEmbedding = self.visitSpecial(node)
        nodeEmbedding = 1/hash(node) + 1/hash(nodeEmbedding) * 0.005
        if node not in self.nodes:
            self.nodes.append(node)
            self.hashedNodes.append(nodeEmbedding)

        if isinstance(node, ast.AST):
            for child in list(ast.iter_child_nodes(node)):
                childEmbedding = self.visitSpecial(child)
                childEmbedding = 1/hash(child) + 1/hash(childEmbedding) * 0.005
                self.edges.append([nodeEmbedding, childEmbedding])
                if child not in self.nodes:
                    self.nodes.append(child)
                    self.hashedNodes.append(childEmbedding)
                    self.generic_visit(child)

        elif isinstance(node, list):
            for child in list(ast.iter_child_nodes(node)):
                childEmbedding = self.visitSpecial(child)
                childEmbedding = 1/hash(child) + 1/hash(childEmbedding) * 0.005
                self.edges.append([(node, nodeEmbedding), (child, childEmbedding)])
                if child not in self.nodes:
                    self.nodes.append(child)
                    self.hashedNodes.append(childEmbedding)
                    self.generic_visit(child)
