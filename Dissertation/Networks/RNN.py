import tensorflow as tf
from typing import List

class RNN:
    def __init__(self):
        pass

    def __init__(self, layerName: str, x_train: tf.Tensor, y_train: tf.Tensor, x_test: tf.Tensor, y_test: tf.Tensor, activationFunction: str = None, 
    neurons:int = None, dropoutRate: float =  None):
        self.layerName = layerName.lower()

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        if activationFunction is not None:
            self.activationFunction = self.getActivationFunction(activationFunction)

        if neurons is not None:
            self.neurons = neurons

        if dropoutRate is not None:
            self.dropoutRate = dropoutRate

    def RNNLayer(self, neurons: int, activationFunction: str, useBias: bool, inputShape, returnSequences):
        activationFunction = self.getActivationFunction(activationFunction)
        return tf.keras.layers.SimpleRNN(neurons, activation=activationFunction, use_bias=useBias, return_sequences=returnSequences, input_shape=inputShape)

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

    def DenseLayer(self, neurons: int, useBias: bool):
        activationFunction = self.activationFunction
        return tf.keras.layers.Dense(neurons, activationFunction, useBias)

    def getActivationFunction(self, activationFunction: str):
        if activationFunction == 'softmax':
            return tf.nn.softmax
        elif activationFunction == 'relu':
            return tf.nn.relu
        elif activationFunction == 'tanh':
            return tf.tanh
        elif activationFunction == 'sigmoid':
            def logSigmoid(x):
                x = 1.0/(1.0 + tf.math.exp(-x)) 
                return x
            return logSigmoid
        else:
            return None

    def runModel(self, layerType: str, neurons: int, epochs: int, batchSize: int, filename: str = None):
        print("USING THE RECURRENT NEURAL NETWORK")
        inputLayer = tf.keras.layers.InputLayer(input_shape=(self.x_train.shape[1], 1))
        inputShape=(self.x_train.shape[0], )

        dropout = self.DropoutLayer(0.3)
        output = self.DenseLayer(2, False)
    
        print("USING", layerType.upper(), "LAYERS")
        model = tf.keras.models.Sequential()
        model.add(inputLayer)
        if layerType == "lstm":
            lstmLayer = self.LSTMLayer(neurons, self.activationFunction, True, inputShape, True)
            model.add(tf.keras.layers.Bidirectional(lstmLayer))
            model.add(tf.keras.layers.LSTM(256))
        elif layerType == "gru":
            gruLayer = self.GRULayer(neurons, self.activationFunction, False, inputShape, True)
            model.add(tf.keras.layers.Bidirectional(gruLayer))
            model.add(tf.keras.layers.GRU(10))
        elif layerType =="rnn":
            rnnLayer = self.RNNLayer(neurons, self.activationFunction, False, inputShape, True)
            model.add(tf.keras.layers.Bidirectional(rnnLayer))
            model.add(tf.keras.layers.SimpleRNN(256))
            
        model.add(output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batchSize, validation_data=(self.x_test, self.y_test))
        if filename is not None:
            model.save(filename)