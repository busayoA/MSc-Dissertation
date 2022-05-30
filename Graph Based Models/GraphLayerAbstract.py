import tensorflow as tf
from abc import ABC

class AbstractGraphLayer(ABC):
    def __init__(self, layers: list, epochs, learningRate, dropout=None, *args, **kwargs):
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

