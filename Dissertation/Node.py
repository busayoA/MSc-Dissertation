class Node():
    def __init__(self):
        self.children = []

    def preOrderTraversal(self, root):
        objectTree = []
        if root is not None:
            objectTree.append(root)
            for i in range(len(root.children)):
                objectTree = objectTree + self.preOrderTraversal(root.children[i])
        return list(set(objectTree))

def createTreeFromEdges(edges):
    nodeEdges = set(i for edge in edges for i in edge)
    nodes = {i: Node() for i in nodeEdges}

    for parent, child in edges:
        if child not in nodes[parent].children:
            nodes[parent].children.append(nodes[child])
        if child in nodeEdges:
            nodeEdges.remove(child)
    
    for edge in nodeEdges:
        return nodes[edge]

