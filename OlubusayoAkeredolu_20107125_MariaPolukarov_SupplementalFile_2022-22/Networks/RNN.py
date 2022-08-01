import tensorflow as tf

class RNN:
    def __init__(self, layerName: str, x_train: tf.Tensor, y_train: tf.Tensor, 
    x_test: tf.Tensor, y_test: tf.Tensor, activationFunction: str = None, 
    neurons:int = None, dropoutRate: float =  None):
        """The RNN class from where all RNN layers are called
        layerName: str - The string representation of the type of layer that has been selected
        x_train: tf.Tensor - The training data
        y_train: tf.Tensor - The training data labels
        x_test: tf.Tensor - The testing data 
        y_test: tf.Tensor - The testing data labels
        activationFunction: str = None - The activation function as a string
        neurons:int = None - The number of units each layer is to have
        dropoutRate: float =  None - The dropout rate"""
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
        """Get the Simple RNN layer"""
        activationFunction = self.getActivationFunction(activationFunction)
        return tf.keras.layers.SimpleRNN(neurons, activation=activationFunction, use_bias=useBias, return_sequences=returnSequences, input_shape=inputShape)

    def LSTMLayer(self, neurons: int, activationFunction: str, useBias: bool, inputShape, returnSequences):
        """Get the LSTM layer"""
        activationFunction = self.getActivationFunction(activationFunction)
        return tf.keras.layers.LSTM(neurons, activation=activationFunction, use_bias=useBias, return_sequences=returnSequences, input_shape=inputShape)

    def GRULayer(self, neurons: int, activationFunction: str, useBias: bool, inputShape, returnSequences):
        "Get the GRU layer"
        activationFunction = self.getActivationFunction(activationFunction)
        return tf.keras.layers.GRU(neurons, activation=activationFunction, use_bias=useBias, return_sequences=returnSequences, input_shape=inputShape)

    def DropoutLayer(self, dropoutRate: float):
        """Get the Dropout Layer
        dropoutRate: float - The probability of a weight being dropped in the range 0 - 1
        """
        return tf.keras.layers.Dropout(dropoutRate)

    def DenseLayer(self, neurons: int, useBias: bool):
        """Get a densely connected layer to be used as the output layer
        neurons: int - The number of units the layer should have
        useBias: bool - Whether or not to use a bias"""
        activationFunction = self.activationFunction
        return tf.keras.layers.Dense(neurons, activationFunction, useBias)

    def getActivationFunction(self, activationFunction: str):
        """Retrieve the activation function based on the input to the method
        activationFunction: str - A string representation of the activation function to be retrieved"""
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
        """ Run the RNN model
        layerType: str - The type of layer: LSTM, GRU or RNN
        neurons: int - The number of units for the chosen layer type
        epochs: int - THe number of times the data will be passed back and forth in the network
        batchSize: int - The size of each training batch in the model
        filename: str = None - The name of the file to save the model into
        """
        print("USING THE RECURRENT NEURAL NETWORK")

        inputLayer = tf.keras.layers.InputLayer(input_shape=(self.x_train.shape[1], 1))
        inputShape=(self.x_train.shape[0], )

        dropout = self.DropoutLayer(0.3)
        output = self.DenseLayer(2, False)
    
        print("USING", layerType.upper(), "LAYERS")
        model = tf.keras.models.Sequential()
        model.add(inputLayer)
        if layerType == "lstm":
            print("USING LSTM LAYERS")
            lstmLayer = self.LSTMLayer(neurons, self.activationFunction, True, inputShape, True)
            model.add(tf.keras.layers.Bidirectional(lstmLayer))
            model.add(tf.keras.layers.LSTM(256))
        elif layerType == "gru":
            print("USING GRU LAYERS")
            gruLayer = self.GRULayer(neurons, self.activationFunction, False, inputShape, True)
            model.add(tf.keras.layers.Bidirectional(gruLayer))
            model.add(tf.keras.layers.GRU(10))
        elif layerType =="rnn":
            print("USING SRNN LAYERS")
            rnnLayer = self.RNNLayer(neurons, self.activationFunction, False, inputShape, True)
            model.add(tf.keras.layers.Bidirectional(rnnLayer))
            model.add(tf.keras.layers.SimpleRNN(256))
            
        model.add(output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batchSize, validation_data=(self.x_test, self.y_test))
        if filename is not None:
            model.save(filename)