import dgl
import networkx as nx
import readCodeFiles as rcf
import tensorflow as tf

class GraphFFModel(tf.keras.Model):
    def __init__(self, layers: list(str), layerNeurons: list(int), epochs, learningRate, dropout=None, *args, **kwargs):
        super(GraphFFModel, self).__init__(*args, **kwargs)
        self.layers = layers
        self.layerNeurons = layerNeurons
        self.dropout = dropout
        self.layerCount = len(layers)
        self.hiddenLayerCount = len(layers)-2
        self.featureCount = layers[0]
        self.classCount = layers[-1]
        self.epochs = epochs
        self.learningRate = learningRate
        self.parameterCount = 0

        self.w, self.wGradients, self.b, self.bGradients = {}, {}, {}, {}
        

        for i in range(1, self.layerCount):
            self.w[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], self.layers[i-1])))
            self.b[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], 1)))

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        for i in range(1, self.layerCount): 
            x = tf.matmul(x, tf.transpose(self.w[i])) + tf.transpose(self.b[i])
            x = 1.0/(1.0 + tf.math.exp(-x))
        return x



x_train, y_train, x_test, y_test = rcf.readCodeFiles()
x_train_graph = nx.from_numpy_array(x_train)
x_train_dgl = dgl.from_networkx(x_train_graph)

layers = ['Input', 'Perceptron_1', 'Perceptron_2', 'Output']
layerNeurons = [len(x_train[0]), 128, 128, 2]
gb = GraphFFModel(layers, layerNeurons, 3, 0.01)
# x = gb(x_train)
# print(x)
