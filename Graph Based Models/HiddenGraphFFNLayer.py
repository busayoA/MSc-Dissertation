import tensorflow as tf
import networkx as nx
import numpy as np
from BinaryGraphInputLayer import BinaryGraphInputLayer as BGIL

class HiddenGraphFFNLayer():
    def __init__(self, epochs, learningRate):
        self.epochs = epochs
        self.learningRate = learningRate
        self.weights, self.weightErrors, self.bias, self.biasErrors = [], [], [], []


    def forwardPropagate(self, x_train, x_matrix):
        embeddings = list(x_train.nodes)
        adjList = nx.dfs_successors(x_train)

        for i in range(len(embeddings)):
            node = embeddings[i]
            matrix = x_matrix[i]
            x = sum(node * matrix)
            embeddings[i] = x

            for item in adjList:
                if node == item:
                    adjacentNodes = tf.convert_to_tensor(adjList[item], dtype=np.float32)
                    embeddings[i] = sum(embeddings[i] * adjacentNodes)
                    

        embeddings = tf.convert_to_tensor(embeddings, dtype=np.float32)
        x = tf.reshape(embeddings, (1, len(embeddings)))
        print(x)
        self.weights = tf.Variable(tf.random.normal(shape=(len(embeddings), 2)), dtype=np.float32)
        self.bias = tf.Variable(tf.random.normal(shape=(2, 1)), dtype=np.float32)
        x = (tf.matmul(x, self.weights) + tf.transpose(self.bias)) * self.learningRate
        x = 1.0/(1.0 + tf.math.exp(-x)) 

        return x

    def backPropagate(self, x_train, y_train):
        pass

    def lossFunction(self, outputs, y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, outputs))


merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort"
quick = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Quick Sort"

inputLayer = BGIL()
xTrain, yTrain, xTest, yTest = inputLayer.splitTrainTest(merge, quick)
x_train, x_train_matrix, y_train, x_test, x_test_matrix, y_test = inputLayer.getDatasets(xTrain, yTrain, xTest, yTest)



hiddenLayer = HiddenGraphFFNLayer(10, 0.003)
out = hiddenLayer.forwardPropagate(x_train[1], x_train_matrix[1])
print(out)
# loss = hiddenLayer.lossFunction(out, y_train[0])
# print(loss.numpy())