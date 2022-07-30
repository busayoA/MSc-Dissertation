import os, ast
from ParsingAndEmbeddingLayers.Trees.TreeNode import TreeNode
from ParsingAndEmbeddingLayers.Visitor import Visitor, HashVisitor

class TreeParser():
    def __init__(self, hashed: bool):
        """
        The Tree Parser class where the files are parsed for processing into trees

        hashed: bool - Whether or not hashing is to be used        
        """
        self.hashed = hashed

    def convertToTree(self, filePath):
        """
        Convert the contents of a file into a tree

        filePath - The file who's contents are to be converted into a tree data structure

        Returns
        tree - The tree representation of the contents of 'filePath'
        """
        programAST = '' #placeholder for the file contents
        with open (filePath, "r") as file:
            programAST = ast.parse(file.read())

        if self.hashed is True: #if we're working with hashed data, use the HashVisitor class
            visitor = HashVisitor()
            visitor.generic_visit(programAST)
            #construct a tree from the edges provided by the HashVisitor
            tree = self.createTreeFromEdges(visitor.edges) 
        else: #if we're working with unhashed data, use the Visitor class
            visitor = Visitor()
            visitor.generic_visit(programAST)
            tree = self.createTreeFromEdges(visitor.edges)
        return tree

    def parse(self, filePath):
        """
        Call the 'convertToTree' method on the contents of a file and assign class labels

        filePath - The file to be converted into a tree

        Returns
        trees - The list of trees from each file
        labels - The list of class labels for each tree
        """
        trees, labels = [], []
        os.chdir(filePath)
        for file in os.listdir():
            # Check whether file is in text format or not
            if file.endswith(".py"):
                path = f"{filePath}/{file}"
                # call read text file function
                tree = self.convertToTree(path)
                trees.append(tree)
                if filePath.find("Merge") != -1:
                    labels.append(0)
                elif filePath.find("Quick") != -1:
                    labels.append(1)
        return trees, labels

    def createTreeFromEdges(self, edges):
        """
        Given a set of tree edges, create an abstract tree

        edges - The edges from which to create a tree
        """
        individualNodes = [] #placeholder for the nodes in the edges list
        for edge in edges: #for each pair of nodes that make an edge
            for i in edge: # for each node in a pair of nodes
                if i not in individualNodes: # prevent duplicate nodes
                    individualNodes.append(i)
        
        # create a set of all the nodes as TreeNode objects
        nodes = {i: TreeNode(i) for i in individualNodes}

        for parent, child in edges: #for each pair of edges
            # make sure the child node is not added to the list of child nodes twice
            if child not in nodes[parent].children: 
                nodes[parent].children.append(nodes[child])
            if child in individualNodes:
                # remove a child from the overall node list once it has been added to its parent's child list
                individualNodes.remove(child)
        for edge in individualNodes:
            return nodes[edge] #return the remaining nodes in the edge list
