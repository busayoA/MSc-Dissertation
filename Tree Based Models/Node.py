import ast
class Node(object):
    def __init__(self, nodeObject):
        self.children = []
        self.childIDs = []
        self.nodeObject = nodeObject
        self.traversed = False
        self.nodeID = self.nodeToString()

    def addChild(self, childNode):
        self.children.append(childNode)
        self.childIDs.append(childNode.nodeToString())

    def printNode(self):
        self.sortConnections()
        print(self.nodeID, [node.nodeID for node in self.children])

    def nodeToString(self):
        s = list(ast.iter_fields(self.nodeObject))
        if len(s) == 0:
            return None
        else:
            return s[0]
        # if isinstance(self.nodeObject, (ast.Module, ast.Interactive, ast.Expression, ast.Try)):
        #     return list(ast.iter_fields(self.nodeObject))
        # elif isinstance(self.nodeObject, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        #     return self.nodeObject.name
        # elif isinstance(self.nodeObject, ast.Return):
        #     return self.nodeObject._fields
        # elif isinstance(self.nodeObject, (ast.Delete, ast.Assign)):
        #     return self.nodeObject._fields
        # elif isinstance(self.nodeObject, (ast.AugAssign, ast.For, ast.AsyncFor)):
        #     return self.nodeObject._fields
        # elif isinstance(self.nodeObject, (ast.While, ast.If, ast.Assert)):
        #     return self.nodeObject._fields
        # elif isinstance(self.nodeObject, (ast.With, ast.AsyncWith)):
        #     return self.nodeObject._fields
        # elif isinstance(self.nodeObject, ast.Raise):
        #     return self.nodeObject._fields
        # elif isinstance(self.nodeObject, (ast.Import, ast.Global, ast.Nonlocal)):
        #     return self.nodeObject._fields
        # elif isinstance(self.nodeObject, ast.ImportFrom):
        #     return self.nodeObject._fields
        

    def sortConnections(self):
        self.children.sort(key=lambda node: node.nodeID)