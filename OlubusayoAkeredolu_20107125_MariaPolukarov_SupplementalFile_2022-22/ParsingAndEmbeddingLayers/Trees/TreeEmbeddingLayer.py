import random
import tensorflow as tf
from typing import List

class TreeEmbeddingLayer():
    def __init__(self, values: list[List, List]):
        """
        The stage where the nodes in each tree are vectorized/embedded
        values: list[List, List] - A list containing the trees and their class labels
        """
        self.root = values[0] #get the root node
        self.nodes =  self.root.preOrderTraversal(self.root) #get all the nodes in the tree
        self.label = values[1] #the class label
        self.rootVec = random.random() #set the root node's embedding to a random float
        self.weights = {}
        self.bias = {}
        self.vectorEmbeddings = [[self.root, self.rootVec]]
        self.vectors = [self.rootVec] #the list of vectorized embeddings
        self.treeDepth = self.getTreeDepth(self.root) #the tree depth
        self.unVectorised = self.root.preOrderTraversal(self.root) #all the non-root nodes are currently unvectorized
        self.rootIndex = self.nodes.index(self.root) #the index of the root node
        #the root node has been vectorized so remove it from the unvectorized list
        self.unVectorised.remove(self.root) 
       
        self.initialiseInputWeights()
        self.embeddingFunction(self.root, None)

    def getTreeDepth(self, root):
        """
        Get the depth of the tree
        The depth is the maximum length of a single branch in the tree
        """
        if root is None:
            # return 0 is the root is empty 
            return 0
        maxDepth = 0
        for child in root.children:
            # recursively loop through the tree to find the longest branch
            maxDepth = max(maxDepth, self.getTreeDepth(child))   
        return maxDepth + 1

    def initialiseInputWeights(self):
        """
        Initialise a set of random weights and biases for each node to be 
        used in calculating the final vectorized form of the node
        """
        for i in range(len(self.nodes)):
            self.weights[i] = tf.Variable(tf.random.normal(shape=(1, 1)))
            self.bias[i] = tf.Variable(tf.random.normal(shape=(1, 1)))
    
    def embeddingFunction(self, node, parent):
        """
        A recursive function that calls the vectorization function of individual nodes in the tree

        node - The current node to be embedded
        parent - The parent node of 'node'

        Returns
        self.vectors - The list of all the vecotirzed nodes in the tree
        """
        if len(self.unVectorised) == 0:
            return self.vectors
        
        if parent is None: #when working with the root node
            functionNodes = node.children  #all the functions in the program file
            for function in functionNodes:
                if function in self.unVectorised: #while there are still unvectorized function nodes
                    self.unVectorised.remove(function)
                    rootIndex = self.nodes.index(self.root) #the index of the root/parent node
                    functionIndex = self.nodes.index(function) #the index of the current node

                    # call the vectorization function on the function node and add it to the vectors list
                    vec = self.vecFunction(len(functionNodes), rootIndex, function, functionIndex)
                    self.vectors.append(vec)
                    self.vectorEmbeddings.append([function, vec])

                    # repeat for all the children of the current function node
                    for child in function.children:
                        self.embeddingFunction(child, function)
        else:
            # When working with child nodes deeper into the tree
            if node in self.unVectorised:
                self.unVectorised.remove(node)
                parentIndex = self.nodes.index(parent)
                childIndex = self.nodes.index(node)
                vec = self.vecFunction(len(parent.children), parentIndex, node, childIndex)
                self.vectors.append(vec)
                self.vectorEmbeddings.append([node, vec])
                for child in node.children:
                    self.embeddingFunction(child, node)
        
    def vecFunction(self, parentChildCount, parentIndex, child, index):
        """
        The vectorization function where 'unhashing' is carried out
        
        parentChildCount - The number of children the current node's parent has
        parentIndex - The index of the current node's parent in the node list
        child - The current node to be vectorized
        index - The index of 'child' in the node list

        Returns
        result.numpy() - The vectorized form of the current node
        """
        childCount = len(child.children) #the number of children the current node has
        pre = 0.0
        if childCount > 0: #if and only if the current node has any children
            pre = float(self.treeDepth) * (parentChildCount/childCount) *  (self.weights[parentIndex] + self.weights[index])
        else: #what to do if the current node does not have any children
            pre = float(self.treeDepth) *  (self.weights[parentIndex] + self.weights[index])
        a = pre + self.bias[index]
        result = tf.reduce_logsumexp(a) * 0.1
        if result < 0:
            result = result * -1.0 #convert the result to a positive float
        return result.numpy()

    def findNodeEmbedding(self, node):
        """
        In the list of vector embeddings, find the embedding corresponding to a particular node
        
        node: The node who's embedding is to be found

        Returns
        embedding - The embedding corresponding to 'node'
        """
        count, embedding = 0, 0.0
        for i in self.vectorEmbeddings:
            n = i[0] #the node
            e = i[1] #the embedding
            if n == node:
                embedding = e
            count += 1        
        return embedding

  