import os, ast, random
import tensorflow as tf
from os.path import dirname, join
from typing import List

class Node():
    def __init__(self):
        self.embedding = 0.0
        self.children = []
        self.treeDepth = 0

    def preorderTraversal(self, root):
        objectTree = []
        if root is not None:
            objectTree.append(root)
            for i in range(len(root.children)):
                objectTree = objectTree + self.preorderTraversal(root.children[i])
        return list(set(objectTree))

    def getDirectChildren(self):
        return self.children

    def getTreeDepth(self, root):
        fullTree = self.preorderTraversal(self)

        maxDepth = 1
        while len(fullTree) > 0:
            maxDepth += 1
            root = fullTree[0]
            fullTree.remove(root)
            children = root.children
            # GO DOWN THE LEFT BRANCH
            if len(children) > 1:
                child = children.pop(1)
                leftDepth = self.getTreeDepth(child)
                if leftDepth > maxDepth:
                    maxDepth = leftDepth
            elif len(children) > 0:
                child = children.pop(0)
                rightDepth = self.getTreeDepth(child)
                if rightDepth > maxDepth:
                    maxDepth = rightDepth
            else:
                return maxDepth

        return maxDepth

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


class Visitor(ast.NodeVisitor):
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.adjList = []

    def generic_visit(self, node):
        if node not in self.nodes:
            self.nodes.append(node)

        if isinstance(node, ast.AST):
            for child in list(ast.iter_child_nodes(node)):
                self.edges.append([node, child])
                if child not in self.nodes:
                    self.nodes.append(child)
                    self.generic_visit(child)

        elif isinstance(node, list):
            for child in list(ast.iter_child_nodes(node)):
                self.edges.append([node, child])
                if child not in self.nodes:
                    self.nodes.append(child)
                    self.generic_visit(child)

class Parser():
    def convertToTree(self, filePath):
        programAST = ''
        with open (filePath, "r") as file:
            programAST = ast.parse(file.read())
        visitor = Visitor()
        visitor.generic_visit(programAST)
        tree = createTreeFromEdges(visitor.edges)

        return tree

    def parse(self, filePath):
        filePath = join(current_dir, filePath)
        trees, labels = [], []
        os.chdir(filePath)
        for file in os.listdir():
            # Check whether file is in text format or not
            if file.endswith(".py"):
                path = f"{filePath}/{file}"
                # call read text file function
                trees.append(self.convertToTree(path))
                if filePath.find("Merge") != -1:
                    labels.append(0)
                elif filePath.find("Quick") != -1:
                    labels.append(1)
        return trees, labels

parser = Parser()
current_dir = dirname(__file__)
merge = join(current_dir, "./Data/Merge Sort")
quick = join(current_dir, "./Data/Quick Sort")

mergeTree, mergeLabels = parser.parse(merge)
quickTree, quickLabels = parser.parse(quick)
x = mergeTree + quickTree 
y = mergeLabels + quickLabels 

y = tf.keras.utils.to_categorical(y)
pairs = []
def attachLabels(x, y):
    for index in range(len(x)):
        pairs.append([x[index], y[index]])
    random.shuffle(pairs)
    return pairs

pairs = attachLabels(x, y)
split = int(0.8 * len(pairs))
train, test = pairs[:split], pairs[split:]

class InputLayer():
    def __init__(self):
        self.weights = {}
        self.bias = {}

    def testOnOneTree(self, values: List[List]):
        
        self.root = values[0]
        self.fullTree = self.root.preorderTraversal(self.root)
        for i in range(len(self.fullTree)):
            self.weights[i] = tf.Variable(tf.random.normal(shape=(1, 1)))
            self.bias[i] = tf.Variable(tf.random.normal(shape=(1, 1)))

        label = values[1]

        rootVec = random.random()
        vectorEmbeddings = [rootVec]
        self.treeDepth = self.root.getTreeDepth(self.root)
        if len(self.root.children) > 0:
            child1 = self.root.children[0]
            parentIndex = self.fullTree.index(self.getParentNode(self.fullTree, child1))
            index = self.fullTree.index(child1)
            vec = self.vecFunction(rootVec, parentIndex, child1, self.treeDepth, index)
            vectorEmbeddings.append(vec)

        return vectorEmbeddings 

    def vecFunction(self, parentVec, parentIndex, child, treeDepth, index):
        # my function is tanh(treeDepth * sum(childCount * parentVec * child weights) + bias)
        childCount = len(child.children)

        a = (childCount * parentVec * self.weights[parentIndex] * self.weights[index])
        b = (float(treeDepth) * a) + self.bias[index] 
        result = tf.reduce_logsumexp(b)
        return result.numpy()

    def getParentNode(self, fullTree, child):
        index = 0
        parent = None
        found = False
        while found is False and index < len(fullTree):
            found = True
            currentNode = fullTree[index]
            currentNodeChildren = currentNode.children
            if child in currentNodeChildren:
                found = True
                parent = currentNode
            else:
                found = False
            index += 1
        
        return parent


tbn = InputLayer()
embeddings = tbn.testOnOneTree(train[0])
print()

        