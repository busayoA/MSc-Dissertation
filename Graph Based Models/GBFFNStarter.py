import networkx as nx
import numpy as np
import tensorflow as tf
import readCodeFiles as rcf
import matplotlib as mlp
import matplotlib.pyplot as plt

mlp.use('Qt5Agg')


class GBFFN():
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

        # Set up the model based on the number of layers (minus the input layer):
        for i in range(1, self.layerCount):
            self.weights[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], self.layers[i-1])))
            self.bias[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], 1)))

        for i in range(1, self.layerCount):
            self.parameterCount += self.weights[i].shape[0] * self.weights[i].shape[1]
            self.parameterCount += self.bias[i].shape[0]

        print(self.featureCount, "features,", self.classCount, "classes,", self.parameterCount, "parameters, and", self.hiddenLayerCount, "hidden layers", "\n")
        for i in range(1, self.layerCount-1):
            print('Hidden layer {}:'.format(i), '{} neurons'.format(self.layers[i]))

    def forwardPropagate(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        for i in range(1, self.layerCount): 
            x = tf.matmul(x, tf.transpose(self.weights[i])) + tf.transpose(self.bias[i])
            x = 1.0/(1.0 + tf.math.exp(-x))
        return x

    def backPropagate(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            outputs = self.forwardPropagate(x)
            loss = self.lossFunction(outputs, y)
        for i in range(1, self.layerCount):
            self.weightErrors[i] = tape.gradient(loss, self.weights[i])
            # optimizer.apply_gradients(zip(self.weightErrors[i], self.weights[i]))
            self.biasErrors[i] = tape.gradient(loss, self.bias[i])
            # optimizer.apply_gradients(zip(self.biasErrors[i], self.bias[i]),global_step=tf.compat.v1.train.get_or_create_global_step())

            # print(self.weightErrors[i])
            # print(self.biasErrors[i])
        del tape
        self.updateWeights()
        return loss.numpy()

    def lossFunction(self, x, y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, x))

    def updateWeights(self):
        for i in range(1, self.layerCount):
            self.weights[i].assign_sub(self.learningRate * self.weightErrors[i])
            self.bias[i].assign_sub(self.learningRate * self.biasErrors[i])

    def predict(self, x):
        outputs = self.forwardPropagate(x)
        return tf.argmax(tf.nn.softmax(outputs), axis=1)

    def makePrediction(self, x):
        outputs = self.backPropagate(x)
        return outputs #tf.argmax(tf.nn.softmax(x), axis=1)

    def trainModel(self, xTrain, yTrain, xTest, yTest):
        # xTrain = tf.convert_to_tensor(xTrain, dtype=tf.float32)
        metrics = {'trainingLoss': [], 'accuracy': []}
        loss = 0.

        for i in range(self.epochs):
            
            print('Epoch {}'.format(i), end='........')
            for j in range(len(xTrain)):
                loss = self.backPropagate(xTrain, yTrain)

                metrics['trainingLoss'].append(loss)

            val_preds = self.predict(xTest)
            metrics['accuracy'].append(np.mean(np.argmax(yTest, axis=1) == val_preds.numpy()))
            print('Accuracy:', metrics['accuracy'][-1], 'Loss:', metrics['trainingLoss'][-1])

        return metrics

x_train, y_train, x_test, y_test, x_train_graph, x_test_graph = rcf.readCodeFiles()
x_train_edges, train_node_encodings, train_node_types, train_nodes = [], [], [], []
index = 0

for graph in x_train_graph:
    myList = [graph[edge0][edge1]['hashed'] for edge0, edge1 in graph.edges]
    x_train_edges.append(myList)
    train_nodes.append(list(graph.nodes))

    encodings, types = [], []
    nodeList = graph.nodes(data ='encoding')
    nodeTypes = graph.nodes(data ='nodeType')
    for item, hashed in nodeList:
        encodings.append(hashed)
    train_node_encodings.append(encodings)

    for item, nodeType in nodeTypes:
        types.append(nodeType)
    train_node_types.append(types)


# print(train_node_features)
index_edges = x_train_edges[index]
index_node = train_node_encodings[index]
index_graph = x_train_graph[index]
index_types = train_node_types[index]

graphData = (tf.convert_to_tensor(index_node), tf.convert_to_tensor(index_edges), index_graph, tf.convert_to_tensor(index_types))
print(graphData)
# print(index_graph)
 
# nx.draw_networkx(index_graph)
# plt.show()

# class GBFFN(tf.keras.Model):
    

