import ast
class Node(object):
    def __init__(self, nodeObject):
        self.children = []
        self.childIDs = []
        self.nodeObject = nodeObject
        self.traversed = False

    def addChild(self, childNode):
        self.children.append(childNode)
        self.childIDs.append(childNode.nodeToString())

    def printNode(self):
        self.sortConnections()
        print(self.nodeID, [node.nodeID for node in self.children])

    # def nodeToString(self):
    #     if isinstance(self.nodeObject, (ast.Module, ast.Interactive, ast.Expression)):
    #         return "root"
    #     elif isinstance(self.nodeObject, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
    #         return "" + self.nodeObject.name
    #     elif isinstance(self.nodeObject, (ast.Global, ast.Nonlocal)):
    #         return "" + self.nodeObject.names
    #     elif isinstance(self.nodeObject, (ast.Return)):
    #         return "Return" + self.nodeObject.value
    #     elif isinstance(self.nodeObject, ast.Delete):
    #         return "Delete" + self.nodeObject.targets
    #     elif isinstance(self.nodeObject, ast.Assign):
    #         return "Assign" + "".join(self.nodeObject._fields)
    #     elif isinstance(self.nodeObject, ast.AugAssign):
    #         return "" + self.nodeObject.target + self.nodeObject.op + self.nodeObject.value
    #     elif isinstance(self.nodeObject, ast.AnnAssign):
    #         return "" + self.nodeObject.target + self.nodeObject.annotation + self.nodeObject.value
    #     elif isinstance(self.nodeObject, (ast.For, ast.AsyncFor)):
    #         return "For" + self.nodeObject.target + self.nodeObject.iter, self.nodeObject.body + self.nodeObject.orelse
    #     elif isinstance(self.nodeObject, (ast.If, ast.While, ast.IfExp)):
    #         return "If/While" + "".join(self.nodeObject.body)+ self.nodeObject.orelse
    #     elif isinstance(self.nodeObject, (ast.With, ast.AsyncWith)):
    #         return "With" + self.nodeObject.items + self.nodeObject.body
    #     elif isinstance(self.nodeObject, (ast.Raise)):
    #         return "Raise" + self.nodeObject.exc, "Because" + self.nodeObject.cause
    #     elif isinstance(self.nodeObject, (ast.Try)):
    #         return "Try" + self.nodeObject.body, "Handlers:" + self.nodeObject.handlers + self.nodeObject.orelse + self.nodeObject.finalbody
    #     elif isinstance(self.nodeObject, ast.Assert):
    #         return "Assert" + self.nodeObject.test + self.nodeObject.msg
    #     elif isinstance(self.nodeObject, ast.Import):
    #         return "Import"+self.nodeObject.names
    #     elif isinstance(self.nodeObject, ast.ImportFrom):
    #         return "Import" + self.nodeObject.module + "From"+self.nodeObject.names
    #     elif isinstance(self.nodeObject, ast.Expr):
    #         return "Expression:" + self.nodeObject.value
    #     elif isinstance(self.nodeObject, (ast.Pass, ast.Break, ast.Continue)):
    #         return "Pass/Break/Continue"
    #     elif isinstance(self.nodeObject, (ast.BoolOp)):
    #         return "Boolean" + self.nodeObject.op
    #     elif isinstance(self.nodeObject, (ast.BinOp)):
    #         return self.nodeObject.left + self.nodeObject.op + self.nodeObject.right
    #     elif isinstance(self.nodeObject, (ast.UnaryOp)):
    #         return self.nodeObject.op + self.nodeObject.operand
    #     elif isinstance(self.nodeObject, (ast.Lambda)):
    #         return "Lambda" + self.nodeObject.args + self.nodeObject.body
    #     elif isinstance(self.nodeObject, (ast.Dict)):
    #         return "Keys:" + self.nodeObject.keys + "Values:" + self.nodeObject.values
    #     elif isinstance(self.nodeObject, (ast.Set)):
    #         return "Set:" + self.nodeObject.elts
    #     elif isinstance(self.nodeObject, (ast.ListComp)):
    #         return "List Comprehension:" + self.nodeObject.elt + self.nodeObject.generators
    #     elif isinstance(self.nodeObject, (ast.Compare)):
    #         return "Compare" + self.nodeObject.left + "to" + self.nodeObject.comparators
    #     else:
    #         return "To String"