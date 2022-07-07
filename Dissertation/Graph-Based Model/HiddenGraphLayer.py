import tensorflow as tf
import numpy as np
from typing import List
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

class HiddenGraphLayer():
    def __init__(self, layerName: str, trainTestData: List = None, learningRate: float = None, activationFunction: str = None, 
    neurons:int = None, dropoutRate: float =  None, hiddenLayerCount: int = None, hiddenLayerUnits: List[int] = None):
        self.layerName = layerName.lower()
        if trainTestData is not None:
            self.x_train = trainTestData[0]
            self.y_train = trainTestData[1]
            self.x_test = trainTestData[2]
            self.y_test = trainTestData[3]
        if learningRate is not None:
            self.learningRate = learningRate
        
        if activationFunction is not None:
            self.activationFunction = activationFunction

        if neurons is not None:
            self.neurons = neurons

        if dropoutRate is not None:
            self.dropoutRate = dropoutRate

        if hiddenLayerCount is not None:
            self.hiddenLayerCount = hiddenLayerCount

        if hiddenLayerUnits is not None:
            self.hiddenLayerUnits = hiddenLayerUnits

    def chooseModel(self, inputShape: tuple = None, returnSequences:bool = None):
        if self.layerName == "ffn":
            return self.FFLayer(self.hiddenLayerCount, self.hiddenLayerUnits, self.activationFunction)
        elif self.layerName == "lstm":
            return self.LSTMLayer(self.neurons, self.activationFunction, True, inputShape, returnSequences)
        elif self.layerName == "gru":
            return self.GRULayer(self.neurons, self.activationFunction, True, inputShape, returnSequences)
        elif self.layerName == "dropout":
            return self.DropoutLayer(self.dropoutRate)
        elif self.layerName == "sgd":
            return self.SGDClassifier(self.x_train, self.y_train, self.x_test)
        elif self.layerName == "svm":
            return self.SVMClassifier(self.x_train, self.y_train, self.x_test)
        elif self.layerName == "nb":
            return self.nbClassify(self.x_train, self.y_train, self.x_test)
        elif self.layerName == "rf":
            return self.rfClassify(self.x_train, self.y_train, self.x_test)
        elif self.layerName == "output" or self.layerName == "dense":
            return self.DenseLayer(self.neurons, self.activationFunction, True)

    def SGDClassifier(self, x_train, y_train, x_test):
        text_clf = Pipeline([('clf', SGDClassifier())])
        text_clf.fit(x_train, y_train)

        predictions = text_clf.predict(x_test)
        return predictions

    def SVMClassifier(self, x_train, y_train, x_test):
        text_clf = Pipeline([('clf', LinearSVC())])
        text_clf.fit(x_train, y_train)

        predictions = text_clf.predict(x_test)
        return predictions

    def rfClassify(self, x_train, y_train, x_test):
        """A RANDOM FOREST CLASSIFIER"""
        text_clf = Pipeline([('clf', RandomForestClassifier(n_estimators=5))])
        text_clf.fit(x_train, y_train)
        predictions = text_clf.predict(x_test)
        return predictions
    
    def nbClassify(self, x_train, y_train, x_test):
        """A MULTINOMINAL NB CLASSIFIER"""
        # Build a pipeline to simplify the process of creating the vector matrix, transforming to tf-idf and classifying
        text_clf = Pipeline([('clf', MultinomialNB())])
        text_clf.fit(x_train, y_train)
        predictions = text_clf.predict(x_test)
        return predictions

    def RNNLayer(self, neurons: int, activationFunction: str, useBias: bool):
        activationFunction = self.getActivationFunction(activationFunction)
        return tf.keras.layers.SimpleRNN(neurons, activation=activationFunction, use_bias=useBias)

    def LSTMLayer(self, neurons: int, activationFunction: str, useBias: bool, inputShape, returnSequences):
        activationFunction = self.getActivationFunction(activationFunction)
        return tf.keras.layers.LSTM(neurons, activation=activationFunction, use_bias=useBias, return_sequences=returnSequences, input_shape=inputShape)

    def GRULayer(self, neurons: int, activationFunction: str, useBias: bool, inputShape, returnSequences):
        activationFunction = self.getActivationFunction(activationFunction)
        return tf.keras.layers.GRU(neurons, activation=activationFunction, use_bias=useBias, return_sequences=returnSequences, input_shape=inputShape)

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
        return tf.keras.layers.Dense(neurons, activationFunction, useBias)

    def getActivationFunction(self, activationFunction: str):
        if activationFunction == 'softmax':
            return tf.nn.softmax
        elif activationFunction == 'relu':
            return tf.nn.relu
        elif activationFunction == 'tanh':
            return tf.tanh
        elif activationFunction == 'logsigmoid':
            def logSigmoid(x):
                x = 1.0/(1.0 + tf.math.exp(-x)) 
                return x
            return logSigmoid
        else:
            return None

