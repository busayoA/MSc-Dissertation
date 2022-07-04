import os, ast, random
import tensorflow as tf
from os.path import dirname, join
from Node import Node, Visitor, createTreeFromEdges
from typing import List

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


def attachLabels(x, y):
    pairs = []
    for index in range(len(x)):
        pairs.append([x[index], y[index]])
    random.shuffle(pairs)
    return pairs

def getPaddedTrees(trees):
    maxLen = 0
    embeddings = []
    for tree in trees:
        nodeEmbeddings = tree.preOrderTraversal(tree)
        if maxLen < len(nodeEmbeddings):
            maxLen = len(nodeEmbeddings)

    for tree in trees:
        nodeEmbeddings = tree.preOrderTraversal(tree)
        if len(nodeEmbeddings) < maxLen:
            for j in range(maxLen - len(nodeEmbeddings)):
                nodeEmbeddings.append(Node())

        embeddings.append(nodeEmbeddings)

    return embeddings

def getUnpaddedTrees(trees):
    embeddings = []

    for tree in trees:
        nodeEmbeddings = tree.preOrderTraversal(tree)
        embeddings.append(nodeEmbeddings)

    return embeddings


