class TreeNode:
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

