import ast
import re
import networkx as nx

class ASTToGraph(ast.NodeVisitor):
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.adjList = []

    def getClassName(self, node):
        className = '' + node.__class__.__name__
        visitor = getattr(self, className, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        if node not in self.nodes:
            self.nodes.append(node)

        if isinstance(node, ast.AST):
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

    def createAdjList(self):
        for node in self.nodes:
            children = list(ast.iter_child_nodes(node))
            if len(children) > 0:
                self.adjList.append([1/hash(node), [1/hash(child) for child in children]])

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

    def visitModule(self, node: ast.Module):
        return node.body

    def visitDef(self, node: ast.FunctionDef or ast.AsyncFunctionDef or ast.ClassDef):
        return self.splitIdentifier(node.name)

    def visitReturn(self, node: ast.Return):
        returnValue = "return " + node.value
        return returnValue

    def visitDelete(self, node: ast.Delete):
        returnValue = "delete " + node.targets
        return returnValue

    def visitAssign(self, node: ast.Assign):
        returnValue = "assign " + node.value + " to " + node.targets
        return returnValue
    
    def visitAugAssign(self, node: ast.AugAssign or ast.AnnAssign):
        returnValue = "assign " + node.value + " to " + node.target
        return returnValue

    def visitAttribute(self, node: ast.Attribute):
        returnValue = node.attr + " = " + node.value
        return returnValue

    def visitName(self, node: ast.Name):
        return node.id
    
    def visitConstant(self, node: ast.Constant):
        return node.value

    


merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort/1.py"
def readAST():
    with open (merge, "r") as file:
        return ast.parse(file.read())

programAST = readAST()


node = ASTToGraph()
body = node.visitModule(programAST)
print(node.visitDef(body[1]))
