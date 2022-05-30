import dgl
import networkx as nx
import readCodeFiles as rcf
import tensorflow as tf



x_train, y_train, x_test, y_test = rcf.readCodeFiles()
x_train_graph = nx.from_numpy_array(x_train)
x_train_dgl = dgl.from_networkx(x_train_graph)

class GraphFFLayer(tf.keras.layers.Layer):
    def __init__(self, layers: list, epochs, learningRate, dropout=None, *args, **kwargs):
        super(GraphFFLayer, self).__init__(*args, **kwargs)
        self.layers = layers
        self.dropout = dropout
        self.layerCount = len(layers)
        self.hiddenLayerCount = len(layers)-2
        self.featureCount = layers[0]
        self.classCount = layers[-1]
        self.epochs = epochs
        self.learningRate = learningRate
        self.parameterCount = 0

        self.w, self.wDeltas, self.b, self.bDeltas = {}, {}, {}, {}
        

        for i in range(1, self.layerCount):
            self.w[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], self.layers[i-1])))
            self.b[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], 1)))

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        for i in range(1, self.layerCount): 
            x = tf.matmul(x, tf.transpose(self.w[i])) + tf.transpose(self.b[i])
            x = 1.0/(1.0 + tf.math.exp(-x))
        return x



# layers = [len(x_train[0]), 128, 128, 2]
# gb = GraphFFLayer(layers, 3, 0.01)
# x = gb(x_train)
# print(x)
