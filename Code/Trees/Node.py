class Node():
    def __init__(self, nodeName, contents):
        self.id = nodeName
        self.contents = contents
        print(self.id, self.contents)

n = Node("root", ["swap = true"])