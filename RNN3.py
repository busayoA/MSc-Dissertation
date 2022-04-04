import numpy as np
import tensorflow as tf
from scipy.special import logsumexp
import random
from statistics import mean


from torch import tensor

class RNN:
    def __init__(self, layers, epochs, lr, batchSize, stepsPerEpoch):
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.batchSize = batchSize
        self.stepsPerEpoch = stepsPerEpoch
        self.layerCount = len(layers)
        self.hiddenLayerCount = len(layers)-2
        self.featureCount = layers[0]
        self.classCount = layers[-1]
        self.parameterCount = 0
        self.weights, self.bias = [], []

        # Set up the model based on the number of layers (minus the input layer):
        for i in range(1, self.layerCount):
            self.weights.append([[random.random() for k in range(self.layers[i-1])] for j in range(self.layers[i])])
            self.bias.append([[random.random() for k in range(1)] for j in range(self.layers[i])])

        print(self.featureCount, "features,", self.classCount, "classes", "and", self.hiddenLayerCount, "hidden layers", "\n")

        for i in range(1, self.layerCount-1):
            print('Hidden layer {}:,'.format(i), '{} neurons'.format(self.layers[i]))


    def forwardPass(self, featureValues):
        finalValues = []
        for i in range(self.layerCount-1):
            featureWeights = self.weights[i]
            featureBias = self.bias[i]
            # print(featureBias)
            for j in range(1, self.layerCount):
                for k in range(self.layers[j]):
                    activationValue = mean(featureWeights[j]) + featureBias[j][0]
                    transferValue =  activationValue * (1.0 - activationValue)
                    finalValues = transferValue * featureValues
        return finalValues   

    def backwardPass(self, expectedOutput, actualValues):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            errorList = []
            if i is not len(self.layers)-1:
                for j in range(layer):
                    thisError = 0.0
                    for unit in range(self.layers[i+1]):
                        transferValue =  mean(expectedOutput) * (1.0 - mean(expectedOutput))
                        thisError += transferValue
                    errorList.append(thisError)
            else:
                for j in range(layer):
                    errorList.append(actualValues[j] - expectedOutput[j])
            # for j in range(len(layer)):
            #     unit = layer[j]
            #     unit['delta'] = errorList[j] * self.transferFunction(unit['output'])

    def updateParameters(self, featureValues):
        for i in range(len(self.layers)):
            values = featureValues[:-1]
            if i is not 0:
                values = [unit['output'] for unit in self.layers[i - 1]]
            for unit in self.layers[i]:
                for j in range(len(values)):
                    unit['weights'][j] -= self.lr  * values[j]
                unit['weights'][-1] -= self.lr * unit['delta']

    def trainModel(self, x_train, y_train, x_test, y_test):
        for epoch in range(self.epochs):
            loss = 0
            index = 0
            for features in x_train:
                output = self.forwardPass(features)
                expected = [0] * self.classCount
                expected[y_train[index]] = 1
                loss += sum([(expected[i]-output[i])**2 for i in range(len(expected))])
                self.backwardPass(expected, output)
                # self.updateParameters(row, learningRate)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.lr, loss))

    def predict(self, featureValues):
        outputs = self.forwardPass(featureValues)
        return outputs.index(max(outputs))
        

    # def compute_loss(self, A, Y):
    #     # A = A * (1.0 - A)
    #     # X = tf.convert_to_tensor(X, dtype=tf.float32)
    #     return tf.reduce_mean(tf.nn.softmax(A))
    
    # def update_params(self, lr):
    #     for i in range(1, self.L):
    #         self.W[i].assign_sub(lr * self.dW[i])
    #         self.b[i].assign_sub(lr * self.db[i])

    # def predict(self, X):
    #     A = self.forward_pass(X)
    #     return tf.argmax(tf.nn.relu(A), axis=1)
    
    # def info(self):
    #     num_params = 0
    #     for i in range(1, self.L):
    #         num_params += self.W[i].shape[0] * self.W[i].shape[1]
    #         num_params += self.b[i].shape[0]
    #     print('Input Features:', self.num_features)
    #     print('Number of Classes:', self.num_classes)
    #     print('Hidden Layers:')
    #     print('--------------')
    #     for i in range(1, self.L-1):
    #         print('Layer {}, Units {}'.format(i, self.layers[i]))
    #     print('--------------')
    #     print('Number of parameters:', num_params)

    # def train_on_batch(self, X, Y, lr):
    #     X = tf.convert_to_tensor(X, dtype=tf.float32)
    #     Y = tf.convert_to_tensor(Y, dtype=tf.float32)
        
    #     with tf.GradientTape(persistent=True) as tape:
    #         A = self.forward_pass(X)
    #         loss = self.compute_loss(A, Y)
    #     for key in self.W.keys():
    #         self.dW[key] = tape.gradient(loss, self.W[key])
    #         if self.dW[key] is None:
    #             self.dW[key] = self.W[key] * (1.0 - self.W[key]) * random.randint(0,1)

    #         self.db[key] = tape.gradient(loss, self.b[key])
    #         if self.db[key] is None:
    #             self.db[key] = self.b[key] * (1.0 - self.b[key]) * random.randint(0,1)

    #     del tape
    #     self.update_params(lr)
        
    #     return loss.numpy()

    # def train(self, x_train, y_train, x_test, y_test, epochs, steps_per_epoch, batch_size, lr):
    #     history = {
    #         'val_loss': [],
    #         'train_loss': [],
    #         'val_acc': []
    #     }
        
    #     for e in range(0, epochs):
    #         epoch_train_loss = 0.
    #         print('Epoch {}'.format(e), end='.')
    #         for i in range(0, steps_per_epoch):
    #             x_batch = x_train[i*batch_size:(i+1)*batch_size]
    #             y_batch = y_train[i*batch_size:(i+1)*batch_size]
    #             batch_loss = self.train_on_batch(x_batch, y_batch, lr)
    #             epoch_train_loss += batch_loss
                
    #             if i%int(steps_per_epoch/10) == 0:
    #                 print(end='.')
                    
    #         history['train_loss'].append(epoch_train_loss/steps_per_epoch)

    #         val_A = self.forward_pass(x_test)
    #         history['val_loss'].append(self.compute_loss(val_A, y_test).numpy())
            
            
    #         val_preds = self.predict(x_test)
    #         history['val_acc'].append(np.mean(np.argmax(y_test, axis=1) == val_preds.numpy()))
            
    #         print('Val Acc:', history['val_acc'][-1], 'Validation Loss:', history['val_loss'][-1])


        
    #     return history