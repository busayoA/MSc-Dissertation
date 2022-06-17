import dgl, torch
import tensorflow as tf
import numpy as np
import createDatasets as cd
from dgl.dataloading import GraphDataLoader


class FeedForwardNetwork:
    def __init__(self, layers, epochs, learningRate):
        self.layers = layers
        self.layerCount = len(layers)
        self.hiddenLayerCount = len(layers)-2
        self.featureCount = layers[0]
        self.classCount = layers[-1]
        self.epochs = epochs
        self.learningRate = learningRate
        self.parameterCount = 0
        self.weights, self.weightErrors, self.bias, self.biasErrors = {}, {}, {}, {}

    def forwardPropagate(self, x_train):
        nodes = x_train[0].ndata['encoding']
        labels = x_train[1]
        x = np.reshape(nodes, (1, len(nodes)))
        self.layers[0] = len(nodes)
        for i in range(1, self.layerCount):
            self.weights[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], self.layers[i-1])))
            self.bias[i] = tf.Variable(tf.random.normal(shape=( self.layers[i], 1)))
            x = tf.matmul(x, tf.transpose(self.weights[i])) + tf.transpose(self.bias[i])
            x = 1.0/(1.0 + tf.math.exp(-x))
        
        return x
            # print(self.weights[i])

    def trainModel(self, x_train, y_train, x_test, y_test):
        metrics = {'trainingLoss': [], 'accuracy': []}
        loss = 0.


        for i in range(self.epochs):
            print('Epoch {}'.format(i), end='........')
            # loss = self.backPropagate(xTrain, yTrain)

            metrics['trainingLoss'].append(loss)

            # val_preds = self.predict(xTest)
            # metrics['accuracy'].append(np.mean(np.argmax(yTest, axis=1) == val_preds.numpy()))
            print('Accuracy:', metrics['accuracy'][-1], 'Loss:', metrics['trainingLoss'][-1])

        return metrics
# batched_graph, label = batch
# print('Number of nodes for each graph element in the batch:', batched_graph.batch_num_nodes())
# print('Number of edges for each graph element in the batch:', batched_graph.batch_num_edges())

# graphs = dgl.unbatch(batched_graph)
# print('The original graphs in the minibatch:')
# print(graphs)

x_train = cd.trainingData
x_test = cd.testData

ffn = FeedForwardNetwork([158, 128, 128, 2], 10, 0.001)
ffn.forwardPropagate(x_train[3])
# train_dataloader = GraphDataLoader(x_train, batch_size=5, drop_last=False)
# test_dataloader = GraphDataLoader(x_test, batch_size=5, drop_last=False)

# it = iter(train_dataloader)
# batch = next(it)
# print(batch)
