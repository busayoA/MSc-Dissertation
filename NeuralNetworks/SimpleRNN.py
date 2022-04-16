import tensorflow as tf
import numpy as np
import readTextFiles as rtf
import readCodeFiles as rcf
from sklearn.feature_extraction.text import CountVectorizer
from scipy.special import logsumexp

class RNN:
    def __init__(self, layers, epochs, stepsPerEpoch, batchSize, learningRate):
        self.layers = layers
        self.layerCount = len(layers)
        self.hiddenLayerCount = len(layers)-2
        self.featureCount = layers[0]
        self.classCount = layers[-1]
        self.epochs = epochs
        self.stepsPerEpoch = stepsPerEpoch
        self.learningRate = learningRate
        self.batchSize = batchSize
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

        # for i in range(1, self.layerCount-1):
        #     print(self.weights[i])

    def forwardPropagate(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        for i in range(1, self.layerCount): 
            x = tf.matmul(x, tf.transpose(self.weights[i])) + tf.transpose(self.bias[i])
            # for row in x:
            #     for value in row:
            #         value = value.numpy()
            #         value = 1.0/(1.0 +
            #         value = value/255.
            print(x)
        return x

    def getDerivative(self, values):
        for val in values:
            val = val * (1.0 - val)
        return values

    def calculateLoss(self, x, y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, x))

    def updateWeights(self):
        for i in range(1, self.layerCount):
            self.weights[i] = self.learningRate * self.weightErrors[i] * self.getDerivative(self.weights[i])
            self.bias[i] = self.learningRate * self.biasErrors[i]


    def run(self, x_train, y_train, x_test, y_test):
        history = {'Loss': [], 'Accuracy': []}
        outputs = x_train
        for i in range(self.epochs):
            trainingLoss = 0.
            print('Epoch {}'.format(i), end='.')
            outputs = self.forwardPropagate(x_train)
            y = tf.convert_to_tensor(y_train, dtype=tf.float32)
            loss = self.calculateLoss(outputs, y)

            for i in range(1, len(self.weights)+1):
                self.weightErrors[i] = tf.multiply(tf.multiply(loss, self.weights[i]), self.getDerivative(self.weights[i]))
                self.biasErrors[i] = tf.multiply(tf.multiply(loss, self.bias[i]), self.getDerivative(self.bias[i]))

            self.updateWeights()
            print(end='.')
            trainingLoss += loss.numpy()

            history['Loss'].append(trainingLoss)
            
            val_preds = self.predict(x_test)
            history['Accuracy'].append(np.mean(np.argmax(y_test, axis=1) == val_preds.numpy()))
            print('Accuracy:', history['Accuracy'][-1], 'Loss:', history['Loss'][-1])
        return history

    def predict(self, X):
        A = self.forwardPropagate(X)
        return tf.argmax(tf.nn.softmax(A), axis=1)

x_train, y_train, x_test, y_test = rtf.getData()

x = x_train + x_test
y = y_train + y_test
# print(train)
split = int(0.5*len(x))
x_train = x[:split]
x_test = x[split:]

y_train = y[:split]
y_test = y[split:]

vocab = sorted(set(x_train))

# Vectorize the training and testing data
vectorizer = CountVectorizer(vocabulary=vocab)
vectorizer.fit(x_train)
x_train = vectorizer.transform(x_train)
x_train = x_train.toarray()
# x_train = np.reshape(x_train, (x_train.shape[0], len(x_train[0])))  
x_test  = vectorizer.transform(x_test)
x_test = x_test.toarray()
# x_test =  np.reshape(x_test, (x_test.shape[0], len(x_test[0]))) 
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

epochs = 2
lr = 0.2
batchSize = 100
stepsPerEpoch = 3
rnn = RNN([len(x_train[0]), 20, 20, 5], epochs, stepsPerEpoch, batchSize, lr)
rnn.run(x_train, y_train, x_test, y_test)
        