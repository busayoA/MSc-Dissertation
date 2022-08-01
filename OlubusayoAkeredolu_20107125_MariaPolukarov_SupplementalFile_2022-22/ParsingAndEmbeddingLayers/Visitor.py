import ast, re
import networkx as nx
from abc import ABC, abstractmethod

class AbstractVisitor(ast.NodeVisitor, ABC):
    def __init__(self):
        """
        An Abstract Visitor object. This is the super class that defines the 
        core functionalities of the Visitor and HashVisitor classes
        """
        self.nodes = []
        self.edges = []
        self.adjList = []
        self.hashedNodes = []

    @abstractmethod
    def generic_visit(self, node):
        raise NotImplementedError()

    def splitCamelCase(self, identifier: str):
        """
        Split an identifier based on 'camelCase'

        identifier: str - The identifier to split

        Returns 
        The split identifier
        """
        splitIdentifier = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [i.group(0) for i in splitIdentifier]

    def splitSnakeCase(self, identifier: str):
        """
        Split an identifier based on 'snake_case'

        identifier: str - The identifier to split

        Returns 
        The split identifier
        """
        return identifier.split("_")

    def splitIdentifier(self, identifier):
        """
        Split an idenfifier regardless of whether it is in 'snake_case' or 'camelCase'

        identifier - The identifier to split

        Returns 
        finalSplitID - The split identifier as a list of its parts     
        """
        splitId = self.splitSnakeCase(identifier) #start by splitting based on snake_case
        finalSplitID = []
        idParts = []
        for part in splitId:
            #if there are still more parts to split, split based on camelCase
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
        """
        Convert the contents of an AST into a NetworkX DiGraph object

        Returns
        graph - The NetworkX DiGraph reepresentation of the AST
        """
        graph = nx.DiGraph()
        graph.add_edges_from(self.edges)
        return graph

    def createAdjList(self):
        """
        Create an adjacency list containing all the nodes in the tree using hashing
        """
        for node in self.nodes:
            children = list(ast.iter_child_nodes(node))
            if len(children) > 0:
                self.adjList.append([1/hash(node), [1/hash(child) for child in children]])

    def visitSpecial(self, node):
        """
        Visit a node and return a string representation of the node
        node - The node to visit

        Returns 
        A string representation of 'node'
        """
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
        """
        The special visitor class for Function and Class definition objects

        Returns
        The name of the function or class definition
        """
        return str(node.name)

    def visitReturn(self, node: ast.Return):
        """
        The special visitor class for Return objects

        Returns
        returnValue - The string representation of the value to be returned
        """
        returnValue = "return " + str(node.value)
        return returnValue

    def visitDelete(self, node: ast.Delete):
        """
        The special visitor class for Delete objects

        Returns
        returnValue - The string representation of the value to be deleted
        """
        returnValue = "delete " + str(node.targets)
        return returnValue

    def visitAssign(self, node: ast.Assign):
        """
        The special visitor class for Assign objects

        Returns
        returnValue - The string representation of the value to be assigned and its targets
        """
        returnValue = "assign " + str(node.value) + " to " + str(node.targets)
        return returnValue

    def visitAugAssign(self, node: ast.AugAssign or ast.AnnAssign):
        """
        The special visitor class for Augmented and Annotated Assign objects

        Returns
        returnValue - The string representation of the value to be assigned and its targets
        """
        returnValue = "assign " + str(node.value) + " to " + str(node.target)
        return returnValue

    def visitAttribute(self, node: ast.Attribute):
        """
        The special visitor class for Attribute objects

        Returns
        returnValue - The string representation of the attribute name and its value
        """
        returnValue = str(node.attr) + " = " + str(node.value)
        return returnValue

    def visitName(self, node: ast.Name):
        """
        The special visitor class for Name objects

        Returns
        The name of the object
        """
        return str(node)

    def visitConstant(self, node: ast.Constant):
        """
        The special visitor class for Constant objects

        Returns
        The value of the constant
        """
        return "value = " + str(node.value)




class Visitor(AbstractVisitor):
    def __init__(self):
        """
        The Visitor object visits all the nodes in an AST without hashing them
        """
        super().__init__()

    def generic_visit(self, node):
        """
        Recursively visit each node in the AST and add it to the node list
        """
        if node not in self.nodes:
            self.nodes.append(node)

        if isinstance(node, ast.AST):
            for child in list(ast.iter_child_nodes(node)): #loop through all the children of the current node
                for child in list(ast.iter_child_nodes(node)):
                    self.edges.append([node, child])
                    if child not in self.nodes:
                        self.nodes.append(child)
                        self.generic_visit(child)

        elif isinstance(node, list):
            for child in list(ast.iter_child_nodes(node)):
                self.edges.append([node, child])
                if child not in self.nodes: #prevent duplicate nodes in the node list
                    self.nodes.append(child)
                    self.generic_visit(child) #recursively visit all the nodes


class HashVisitor(AbstractVisitor):
    def __init__(self):
        """
        The HashVisitor object visits all the nodes in an AST and performs hashing on them
        """
        super().__init__()

    def generic_visit(self, node):
        """
        Recursively visit each node in the AST,
        visit that node specially,
        perform the hashing algorithm on it
        add it to the node list
        and add all its children in pairs to the list of edges
        """
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
