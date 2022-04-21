import string, nltk, random
import readFiles as rf
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.special import logsumexp
from nltk.corpus import twitter_samples

class FeedForwardNetwork:
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
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learningRate)
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

    def trainModel(self, xTrain, yTrain, xTest, yTest):
        metrics = {'trainingLoss': [], 'accuracy': []}
        loss = 0.

        for i in range(self.epochs):
            print('Epoch {}'.format(i), end='........')
            index = 0
            loss = self.backPropagate(xTrain, yTrain)
            
            metrics['trainingLoss'].append(loss)
            
            val_preds = self.predict(xTest)
            metrics['accuracy'].append(np.mean(np.argmax(yTest, axis=1) == val_preds.numpy()))
            print('Accuracy:', metrics['accuracy'][-1], 'Loss:', metrics['trainingLoss'][-1])
    
        return metrics

x_train, y_train, x_test, y_test = rf.getVectorizedCodeData()

epochs = 10
lr = 0.001
rnn = FeedForwardNetwork([len(x_train[0]), 128, 128, 3], epochs, lr)

# print(rnn.backPropagate(x_train, y_train))
# print(rnn.predict(x_test, y_test))
metrics = rnn.trainModel(x_train, y_train, x_test, y_test)
print("Average loss:", np.average(metrics['trainingLoss']), "Average accuracy:", np.average(metrics['accuracy']))

