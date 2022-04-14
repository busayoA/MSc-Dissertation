from math import exp
from cv2 import error
import numpy as np, tensorflow as tf
import sklearn
import random
from statistics import mean

class RNN:
    def __init__(self, layers):
        self.layers = layers
        self.L = len(layers)
        self.num_features = layers[0]
        self.num_classes = layers[-1]
        
        self.W = {}
        self.b = {}
        
        self.dW = {}
        self.db = {}

        for i in range(1, self.L):
            self.W[i] = [[random.random() for y in range(self.layers[i])] for x in range(self.layers[i])]
            self.b[i] = [[random.random() for y in range(self.layers[i])] for x in range(self.layers[i])]
        
    def forward_pass(self, x_train):   
        for feature in x_train:
            inputs = feature
            for i in range(1, self.L):
                returnValues = []
                currentLayer = self.layers[i]
                # inputs = (np.transpose(inputs) * self.W[i])  #+ self.b[i]
                index = 0
                for j in range(currentLayer):
                    weight = self.W[i][j][index]
                    bias = self.b[i][j][index]
                    weight = (weight * inputs) + bias
                    returnValues.append(weight)
                inputs = np.array(returnValues)/255.
        return inputs

    def compute_loss(self, A, Y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.transpose(Y), tf.transpose(A)))
    
    def updateWeights(self, lr):
        for i in range(1, self.L):
            weights = np.array(self.W[i])
            self.W[i] = weights - (lr * weights)
            self.b[i] = self.b[i] - (lr * self.b[i])

    def predict(self, featureValues):
         outputs = self.forward_pass(featureValues)
         return np.where(outputs == np.amax(outputs))

    def train(self, x_train, y_train, x_test, y_test, epochs, steps_per_epoch, batch_size, lr):
        history = {
            'val_loss': [],
            'train_loss': [],
            'val_acc': []
        }
        
        for e in range(0, epochs):
            epoch_train_loss = 0.
            print('Epoch {}'.format(e), end='.')
            for i in range(5):
                outputs = self.forward_pass(x_train[:5])
                print(end='.')
                # print(outputs, end="\n\n")

            # self.updateWeights(lr)

            validationAccuracy = 0
            for i in range(len(x_test[:5])):
                prediction = self.predict(x_test[:5])
                if (prediction[0][0] == y_test[i]):
                    validationAccuracy += 1
            print("Accuracy:", validationAccuracy/len(x_test))
            # history['train_loss'].append(epoch_train_loss/steps_per_epoch)
            
            # # x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
            # val_A = self.forward_pass(x_test)
            # history['val_loss'].append(self.compute_loss(val_A, y_test).numpy())
            
            # val_preds = self.predict(x_test)
            # history['val_acc'].append(np.mean(np.argmax(y_test) == val_preds.numpy()))
            # print('Val Acc:', history['val_acc'][-1], 'Loss:', history['train_loss'][-1])
        return history