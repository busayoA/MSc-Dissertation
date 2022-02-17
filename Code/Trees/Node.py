from anytree import AnyNode

class Node(AnyNode):
    def __init__(self, nodeName, declarations):
        self.id = nodeName
        self.declarations = declarations
        print(self.id, self.declarations)

n = Node("root", ["swap = true"])