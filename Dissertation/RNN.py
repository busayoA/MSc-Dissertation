import tensorflow as tf
from typing import List

class RNN():
    def __init__(self):
        pass

    def __init__(self, layerName: str, learningRate: float = None, activationFunction: str = None, 
    neurons:int = None, dropoutRate: float =  None, hiddenLayerCount: int = None, hiddenLayerUnits: List[int] = None):
        self.layerName = layerName.lower()
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
        if self.layerName == "lstm":
            return self.LSTMLayer(self.neurons, self.activationFunction, True, inputShape, returnSequences)
        elif self.layerName == "rnn":
            return self.RNNLayer(self.neurons, self.activationFunction, True)
        elif self.layerName == "gru":
            return self.GRULayer(self.neurons, self.activationFunction, True, inputShape, returnSequences)
        elif self.layerName == "dropout":
            return self.DropoutLayer(self.dropoutRate)
        elif self.layerName == "output" or self.layerName == "dense":
            return self.DenseLayer(self.neurons, self.activationFunction, True)

    def RNNLayer(self, neurons: int, activationFunction: str, useBias: bool):
        activationFunction = self.getActivationFunction(activationFunction)
        return tf.keras.layers.SimpleRNN(neurons, activation=activationFunction, use_bias=useBias)

    def LSTMLayer(self, neurons: int, activationFunction: str, useBias: bool, inputShape, returnSequences):
        activationFunction = self.getActivationFunction(activationFunction)
        return tf.keras.layers.LSTM(neurons, activation=activationFunction, use_bias=useBias, return_sequences=returnSequences, input_shape=inputShape)

    def GRULayer(self, neurons: int, activationFunction: str, useBias: bool, inputShape, returnSequences):
        activationFunction = self.getActivationFunction(activationFunction)
        return tf.keras.layers.GRU(neurons, activation=activationFunction, use_bias=useBias, return_sequences=returnSequences, input_shape=inputShape)

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

