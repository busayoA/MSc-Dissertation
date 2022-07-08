class TreeNode():
    def __init__(self, embedding: float):
        self.children = []
        self.embedding = embedding

    def preOrderTraversal(self, root):
        objectTree = []

        if self.embedding == 0.0:
            if root is not None:
                objectTree.append(root)
                for i in range(len(root.children)):
                    objectTree = objectTree + self.preOrderTraversal(root.children[i])
            return list(set(objectTree))
        else:
            if root is not None:
                objectTree.append(root)
                for i in range(len(root.children)):
                    objectTree = objectTree + self.preOrderTraversal(root.children[i])
            return list(set(objectTree))
    
    def getTreeEmbeddings(self, root):
        fullTree = self.preOrderTraversal(root)
        embeddings = []
        for i in fullTree:
            embeddings.append(i.embedding)
        
        return embeddings

def createTreeFromEdges(edges):
    individualNodes = []
    for edge in edges:
        for i in edge:
            if i not in individualNodes:
                individualNodes.append(i)
    nodes = {i: TreeNode(i) for i in individualNodes}
    for parent, child in edges:
        if child not in nodes[parent].children:
            nodes[parent].children.append(nodes[child])
        if child in individualNodes:
            individualNodes.remove(child)
    for edge in individualNodes:
        return nodes[edge]
