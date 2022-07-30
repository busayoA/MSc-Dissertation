class TreeNode:
    def __init__(self, embedding: float):
        """
        A class to represent individual nodes in a tree

        embedding: float - The embedding representation of a node
        """
        self.children = [] #placeholder for the child nodes of the current TreeNode object
        self.embedding = embedding

    def preOrderTraversal(self, root):
        """
        Run the pre-order traversal algorithm on a root node to get all the nodes present 
        in a tree as well as their respective children

        root: The root node of the tree (Must be a TreeNode object)
        """
        objectTree = [] #placeholder for all the nodes in the tree
        if root is not None: #while the root node in the recursion is not none
            objectTree.append(root)
            for i in range(len(root.children)): 
                # recursively add nodes to the objectTree
                objectTree = objectTree + self.preOrderTraversal(root.children[i])
        return list(set(objectTree))
    
    def getTreeEmbeddings(self, root):
        """
        Run preorder traversal on a root node and get the embeddings of all the nodes in the tree

        root: The root node of the tree

        Returns
        embeddings - The list of embeddings of all the nodes in the tree
        """
        fullTree = self.preOrderTraversal(root)
        embeddings = []
        for i in fullTree:
            embeddings.append(i.embedding)
        
        return embeddings

