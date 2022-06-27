import tensorflow as tf
import networkx as nx
import numpy as np
from typing import List
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

class HiddenGraphLayer():
    def __init__(self, learningRate, layerName: str, activationFunction: str, neurons:int, dropoutRate: float =  None, hiddenLayerCount: int = None, hiddenLayerUnits: List[int] =
    None):
        self.learningRate = learningRate
        self.layerName = layerName.lower()
        self.activationFuntion = activationFunction
        self.neurons = neurons
        self.dropoutRate = dropoutRate
        self.hiddenLayerCount = hiddenLayerCount
        self.hiddenLayerUnits = hiddenLayerUnits

    def chooseModel(self):
        if self.layerName == "ffn":
            return self.FFLayer(self.hiddenLayerCount, self.hiddenLayerUnits, self.activationFuntion)
        elif self.layerName == "rnn":
            return self.RNNLayer(self.neurons, self.activationFuntion, True, self.dropoutRate)
        elif self.layerName == "lstm":
            return self.LSTMLayer(self.neurons, self.activationFuntion, True, self.dropoutRate)
        # elif self.layerName == "sgd":
        #     return self.SGDClassifier()
        # elif self.layerName == "svm":
        #     return self.SVMClassifier()
        elif self.layerName == "dropout":
            return self.DropoutLayer(self.dropoutRate)
        elif self.layerName == "output":
            return self.DenseLayer(self.neurons, self.activationFuntion, True)

    def SGDClassifier(trainSet, trainCategories, testSet):
        text_clf = Pipeline([('vect', CountVectorizer(min_df=5)),('tfidf', TfidfTransformer()),('clf', SGDClassifier())])
        text_clf.fit(trainSet, trainCategories)

        predictions = text_clf.predict(testSet)
        return predictions

    def SVMClassifier(trainSet, trainCategories, testSet):
        text_clf = Pipeline([('vect', CountVectorizer(min_df=5)),('tfidf', TfidfTransformer()),('clf', LinearSVC())])
        text_clf.fit(trainSet, trainCategories)

        predictions = text_clf.predict(testSet)
        return predictions

    def RNNLayer(self, neurons: int, activationFunction: str, useBias: bool, dropoutRate: float):
        activationFunction = self.getActivationFunction(activationFunction)
        return tf.keras.layers.SimpleRNN(neurons, activation=activationFunction, use_bias=useBias)

    def LSTMLayer(self, neurons: int, activationFunction: str, useBias: bool, dropoutRate: float):
        activationFunction = self.getActivationFunction(activationFunction)
        return tf.keras.layers.LSTM(neurons, activation=activationFunction, use_bias=useBias)

    def FFLayer(self, hiddenLayerCount: int, hiddenLayerUnits: List[int], activationFunction: str):
        activationFunction = self.getActivationFunction(activationFunction)
        if len(hiddenLayerUnits) != hiddenLayerCount:
            raise Exception("Something is wrong somewhere, check that again")

        self.layers = []
        for i in range(hiddenLayerCount): 
            self.layers.append(self.DenseLayer(hiddenLayerUnits[i], useBias=True, activationFunction=activationFunction))
        return self.layers

    def DropoutLayer(self, dropoutRate):
        if dropoutRate is None:
            dropoutRate = 0.3
        return tf.keras.layers.Dropout(dropoutRate)

    def DenseLayer(self, neurons: int, activationFunction: str, useBias: bool):
        activationFunction = self.getActivationFunction(activationFunction)
        return tf.keras.layers.Dense(neurons, activationFunction, useBias, input_shape = (2, 2))

    def getActivationFunction(self, activationFunction: str):
        if activationFunction == 'softmax':
            return tf.nn.softmax
        elif activationFunction == 'relu':
            return tf.nn.relu
        elif activationFunction == 'tanh':
            return tf.tanh
        elif activationFunction == 'logsigmoid':
            def logSigmoid(x):
                weights = tf.Variable(tf.random.normal(shape=(len(x), 2)), dtype=np.float32)
                bias = tf.Variable(tf.random.normal(shape=(2, 1)), dtype=np.float32)
                x = tf.matmul(x, weights) + tf.transpose(bias)
                x = 1.0/(1.0 + tf.math.exp(-x)) 
                return x
            return logSigmoid
        else:
            return None

