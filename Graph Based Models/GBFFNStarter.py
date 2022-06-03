from matplotlib.pyplot import step
import networkx as nx
import numpy as np
import tensorflow as tf
import readCodeFiles as rcf



# for i in range(stepsPerEpoch):
#     currentXBatch = x_train_graph[i*batchSize:(i+1)*batchSize]
#     currentYBatch = y_train[i*batchSize:(i+1)*batchSize]
#     currentEdges = []

    # if len(currentXBatch) > 1:
    #     print(currentXBatch)
    #     print(currentYBatch)
    #     print("\n\n")

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

# print(x_test.shape)
# x_train = np.asarray(x_train, dtype=object)
# print(x_train.shape)

# x_train = np.reshape(x_train, (1, len(x_train)))

# batchSize = 10
# learningRate = 0.01
# stepsPerEpoch = int(x_train.shape[0]/batchSize)
# epochs = 20
# layers = [len(x_train), 128, 128, 2]
# ffn = GBFFN(layers, epochs, learningRate)
# # print(rnn.predict(x_test, y_test))
# metrics = ffn.trainModel(x_train, y_train, x_test, y_test)
# print("Average loss:", np.average(metrics['trainingLoss']), "Average accuracy:", np.average(metrics['accuracy']))

x_train_edges, train_node_features = [], []
index = 0
for graph in x_train_graph:
    x_train_edges.append(list(graph.edges))
    train_node_features.append(list(graph.nodes.data()))


# print(train_node_features)
index_edges = x_train_edges[index]
index_node_features = train_node_features[index]
index_graph = x_train_graph[index]

graphData = (index_graph, index_edges, index_node_features)
print(graphData)


# class GBFFN(tf.keras.Model):
    

# for graph in x_test_graph:
#     x_test_edges.append(graph.edges)

# print(nodeFeatures)





# model = GCN(len(trainingDataLoader), 2, 2)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# batchSize = 10
# for graph in trainingDataLoader:
#     pred = model(graph, sum(graph.batch_num_nodes()))
#     loss = func.cross_entropy(pred, y_train_labels[i])
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

