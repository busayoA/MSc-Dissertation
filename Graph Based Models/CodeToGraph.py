import ast
from anytree import AnyNode, RenderTree
from ete3 import Tree as tree
from collections import deque

class Tree(ast.NodeVisitor, AnyNode):
    def __init__(self, ID, parent, branches):
        self.ID = ID
        self.parent = parent
        self.node = AnyNode(id=self.ID, parent=self.parent, children=branches)
        self.branches = branches
        
    def printTree(self):
        print(RenderTree(self.node))

merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort/1.py"
def readAST():
    with open (merge, "r") as file:
        return ast.parse(file.read())

programAST = readAST()

class Node(ast.NodeVisitor):
    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
                        print(item)
            elif isinstance(value, ast.AST):
                self.visit(value)
        
    def walk(self, node):
        todo = deque([node])
        while todo:
            node = todo.popleft()
            todo.extend(ast.iter_child_nodes(node))
            return node
        

node = Node()
node.generic_visit(programAST)