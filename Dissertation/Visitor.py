import ast

class Visitor(ast.NodeVisitor):
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.adjList = []

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

class HashVisitor:
    def __init__(self):
        self.nodes = []
        self.hashedNodes = []

    def generic_visit(self, node):
            if node not in self.nodes:
                nodeEmbedding = self.visitSpecial(node)
                nodeEmbedding = 1/hash(node) + 1/hash(nodeEmbedding) * 0.005
                self.nodes.append(node)
                self.hashedNodes.append(nodeEmbedding)

            if isinstance(node, ast.AST):
                for child in list(ast.iter_child_nodes(node)):
                    childEmbedding = self.visitSpecial(child)
                    childEmbedding = 1/hash(child) + 1/hash(childEmbedding) * 0.005

                    if child not in self.nodes:
                        self.nodes.append(child)
                        self.hashedNodes.append(childEmbedding)
                        self.generic_visit(child)

            elif isinstance(node, list):
                for child in list(ast.iter_child_nodes(node)):
                    childEmbedding = self.visitSpecial(child)
                    childEmbedding = 1/hash(child) + 1/hash(childEmbedding) * 0.005
                    if child not in self.nodes:
                        self.nodes.append(child)
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
