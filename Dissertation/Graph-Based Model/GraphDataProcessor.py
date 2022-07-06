import tensorflow as tf
from parseGraphFiles import Parser
from typing import List
from os.path import dirname, join

parser = Parser()
segmentCount = 40
def splitTrainTest(x, matrices, y):
        split = int(0.7 * len(x))

        x_train = x[:split]
        x_train_matrix = matrices[:split]
        y_train = y[:split]

        x_test = x[split:]
        x_test_matrix = matrices[split:]
        y_test = y[split:]
        
        return x_train, x_train_matrix, y_train, x_test, x_test_matrix, y_test

def runSegmentation(nodeEmbeddings: tf.Tensor, numSegments: int):
    segFunc = tf.math.unsorted_segment_sqrt_n(nodeEmbeddings, tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]), num_segments = numSegments)
    return segFunc

def getMaxLen(x):
    maxLen = 0
    for i in x:
        if len(i) > maxLen:
            maxLen = len(i)

    return maxLen

def padGraphs1(x, maxLen):
    length = len(x)
    for i in range(length):
        if len(x[i]) < maxLen:
            padCount = maxLen - len(x[i])

            x[i] = list(x[i])
            for j in range(padCount):
                x[i].append(0.0)

        x[i] = tf.convert_to_tensor(x[i])
        # x[i] = tf.reshape(x[i], (1, len(x[i])))
    return x

def padGraphs2(x, maxLen):
    length = len(x)
    for i in range(length):
        if len(x[i]) < maxLen:
            padCount = maxLen - len(x[i])
            for j in range(padCount):
                x[i].append(0.0)
    return x

def runParser(processor: int):
    x, matrices, labels = parser.readFiles()
    x_train_all, x_train_matrix, y_train, x_test_all, x_test_matrix, y_test = splitTrainTest(x, matrices, labels)

    x_train_nodes = []
    x_train_graph = []

    for i in range(len(x_train_all)):
        x_train_nodes.append(x_train_all[i][0])
        x_train_graph.append(x_train_all[i][1])

    x_test_nodes = []
    x_test_graph = []
    for i in range(len(x_test_all)):
        x_test_nodes.append(x_test_all[i][0])
        x_test_graph.append(x_test_all[i][1])

    if processor == 1:
        return x_train_graph, x_train_matrix, y_train, x_test_graph, x_test_matrix, y_test
    elif processor == 2:
        return x_train_nodes, y_train, x_test_nodes, y_test

def runProcessor1():
    x_train_graph, x_train_matrix, y_train, x_test_graph, x_test_matrix, y_test = runParser(1)
    x_train = parser.prepareData(x_train_graph, x_train_matrix)
    x_test = parser.prepareData(x_test_graph, x_test_matrix)

    total_x = x_train + x_test
    maxLen = getMaxLen(total_x)
    x_train = padGraphs1(x_train, maxLen)
    x_test = padGraphs1(x_test, maxLen)

    x_train = tf.convert_to_tensor(x_train)
    y_train = tf.keras.utils.to_categorical(y_train)
    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.keras.utils.to_categorical(y_test)  

    return x_train, y_train, x_test, y_test

def runProcessor2():
    xTrain, yTrain, xTest, yTest = runParser(2)

    totalX = xTrain + xTest
    maxLen = getMaxLen(totalX)

    xTrain = padGraphs2(xTrain, maxLen)
    xTest = padGraphs2(xTest, maxLen)

    xTrain2, xTest2 = [], []
    for i in xTrain:
        xTrain2.append(i)

    for i in xTest:
        xTest2.append(i)
    return xTrain2, yTrain, xTest2, yTest

def runProcessor3():
    x_train_graph, x_train_matrix, y_train, x_test_graph, x_test_matrix, y_test = runParser(1)
    x_train = parser.prepareData(x_train_graph, x_train_matrix)
    x_test = parser.prepareData(x_test_graph, x_test_matrix)

    x_train2, x_test2= [], []
    for x in x_train:
        x = tf.convert_to_tensor(x)
        x = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
        x = runSegmentation(x, segmentCount)
        x = tf.reshape(x, (len(x[0]), segmentCount))
        x_train2.append(x[0])

    for x in x_test:
        x = tf.convert_to_tensor(x)
        x = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
        x = runSegmentation(x, segmentCount)
        x = tf.reshape(x, (len(x[0]), segmentCount))
        x_test2.append(x[0])

    return x_train2, y_train, x_test2, y_test


def runProcessor4():
    xTrain, yTrain, xTest, yTest = runParser(2)
    x_train2, x_test2 = [], []
    for x in xTrain:
        x = tf.convert_to_tensor(x)
        x = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
        x = runSegmentation(x, segmentCount)
        x = tf.reshape(x, (len(x[0]), segmentCount))
        x_train2.append(x[0])

    for x in xTest:
        x = tf.convert_to_tensor(x)
        x = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x]
        x = runSegmentation(x, segmentCount)
        x = tf.reshape(x, (len(x[0]), segmentCount))
        x_test2.append(x[0])

    xTrain2, xTest2 = [], []
    for i in x_train2:
        xTrain2.append(i)

    for i in x_test2:
        xTest2.append(i)
    return xTrain2, yTrain, xTest2, yTest



