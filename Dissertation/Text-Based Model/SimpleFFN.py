import tensorflow as tf
import numpy as np

class TextBasedModel:
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
            # print(x)
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

    def runFFModel(self, xTrain, yTrain, xTest, yTest):
        metrics = {'trainingLoss': [], 'trainingAccuracy': [], 'validationAccuracy': []}
        index = 0
        
        for i in range(self.epochs):
            print('Epoch {}'.format(i), end='........')
            predictions = []

            if index % 5 == 0:
                print(end=".")
            if index >= len(yTrain):
                index = 0

            # FIRST FORWARD PASS
            loss = self.backPropagate(xTrain, yTrain)
            # SECOND FORWARD PASS/RECURRENT LOOP
            newOutputs = self.forwardPropagate(xTrain)
            pred = tf.argmax(tf.nn.softmax(newOutputs), axis = 1)
            predictions.append(pred)
            index += 1

            predictions = tf.convert_to_tensor(predictions)
            # print(predictions)
            metrics['trainingLoss'].append(loss)
            unseenPredictions = self.predict(xTest)
            metrics['trainingAccuracy'].append(np.mean(np.argmax(yTrain, axis=1) == predictions.numpy()))
            metrics['validationAccuracy'].append(np.mean(np.argmax(yTest, axis=1) == unseenPredictions.numpy()))
            print('\tLoss:', metrics['trainingLoss'][-1], 'Accuracy:', metrics['trainingAccuracy'][-1], 
            'Validation Accuracy:', metrics['validationAccuracy'][-1])
        return metrics

    def runDenseModel(self, x_train, y_train, x_test, y_test):
        filename = "binaryRNNModel.hdf5"

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.InputLayer(input_shape=(x_train.shape[1], )))
        model.add(tf.keras.layers.Dense(128))
        model.add(tf.keras.layers.Dense(64))
        model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()

        model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

        model.save(filename)

    def runRNNModel(self, x_train, y_train, x_test, y_test):
        filename = "binaryRNNModel.hdf5"

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.InputLayer(input_shape=(x_train.shape[1], 1)))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, input_shape=(x_train.shape[0], 1), 
        return_sequences=True)))
        model.add(tf.keras.layers.GRU(256))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()

        model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

        model.save(filename)


