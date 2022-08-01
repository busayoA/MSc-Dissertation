import random
import tensorflow as tf
import networkx as nx
from networkx import DiGraph 

class GraphEmbeddingLayer:
    def __init__(self, graph: DiGraph):
        """
        The Graph Embedding Layer class that carries out vectorization/unhashing on nodes
        graph: DiGraph - The graph who's nodes are to be vectorized
        """
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.nodeCopy = list(graph.nodes)
        self.edges = list(graph.edges)
        self.root = self.nodes[0] 
        self.rootEmbedding = random.random() #set the root node's embedding to a random float
        self.vectors = [self.rootEmbedding]
        self.nodeCopy.remove(self.root) 
        self.unVectorised = self.nodeCopy #create a list of unvectorized nodes

        self.weights = {}
        for i in range(len(self.nodes)):
            #initiailise a set of random weights for each node
            self.weights[i] = tf.Variable(tf.random.normal(shape=(1, 1)))
            
        self.embeddingFunction(self.root, None) 
        
    def embeddingFunction(self, node, parent):
        """
        The embedding function from where the vectorization function is called on each node recursively
        node - The node to be vectorized
        parent - The parent node of 'node'
        """

        # run the recursive method as long as the unvectorized list is not empty
        if len(self.unVectorised) == 0:
            return self.vectors
        
        if parent is None: #if node is the root node
            # get all the children of the current node
            childNodes = nx.dfs_successors(self.graph, node)
            for child in childNodes:
                if child in self.unVectorised: #check that the node has not been vectorized
                    self.unVectorised.remove(child)
                    rootIndex = self.nodes.index(self.root) #index of the root node
                    childIndex = self.nodes.index(child) #index of the current node
                    parentChildCount = len(self.getChildNodes(self.root)) #the number of children the root node has
                    childNodeChildCount = len(self.getChildNodes(child)) # the number of children this child node has

                    # run the vectorization function on the node given the above variables
                    vec = self.vecFunction(parentChildCount, rootIndex, childIndex, childNodeChildCount)
                    self.vectors.append(vec) 
                    for childNode in self.getChildNodes(child):
                        # recursively call the method until all the nodes are vectorized
                        self.embeddingFunction(childNode, child) 
        else: #if the current node is not the root node
            if node in self.unVectorised:
                self.unVectorised.remove(node)
                parentIndex = self.nodes.index(parent) #index of the parent node
                childIndex = self.nodes.index(node)
                parentChildCount = len(self.getChildNodes(parent))
                childNodeChildCount = len(self.getChildNodes(node))
                vec = self.vecFunction(parentChildCount, parentIndex, childIndex, childNodeChildCount)
                self.vectors.append(vec)
                for childNode in self.getChildNodes(node):
                    self.embeddingFunction(childNode, node)
    
    def getChildNodes(self, node):
        """
        Get the all the child nodes of a node
        node - The node who's children are to be collected

        Returns
        children - All the child nodes of 'node'
        """
        edges = self.edges #all the edges in the graph
        children = []
        for edge in edges:
            #if the right side of the edge is the current node, then the left side is a child node
            if edge[0] == node: 
                children.append(edge[1])

        return children
    
    def vecFunction(self, parentChildCount, parentIndex, childIndex, childNodeChildCount):
        """
        The function in which vectorization is carried out
        parentChildCount - The number of children of the current node's parent node
        parentIndex - The index of the current node's parent node in the node list
        childIndex - The index of the current node in the node list
        childNodeChildCount - The number of children the current node has

        Returns 
        result.numpy() - The numpy representation of the final result of vectorization
        """
        pre = 0.0 
        if childNodeChildCount > 0: #if the current node has one or more children
            pre = (parentChildCount/childNodeChildCount) *  (self.weights[parentIndex] + self.weights[childIndex])
        else: #if the current node does not have any children
            pre = (self.weights[parentIndex] + self.weights[childIndex])
        result = tf.reduce_logsumexp(pre) * 0.1
        if result < 0:
            result = result * -1.0
        return result.numpy()
