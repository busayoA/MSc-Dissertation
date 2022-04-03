from tkinter import W
from cv2 import exp, mean
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

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
            self.W[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], self.layers[i-1])))
            self.b[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], 1)))

        self.info()

    def forward_pass(self, A):
        A = tf.convert_to_tensor(A, dtype=tf.float32)
        for i in range(1, self.L):
            activation = A * tf.reduce_mean(self.W[i])
            activation += tf.reduce_mean(self.b[i])
            transfers = 1.0 / 1.0 + activation
            
            Z = transfers * random.randint(0,1)
            if i != self.L-1:
                A = tf.nn.relu(Z)
            else:
                A = Z
        return A

    def compute_loss(self, A, Y):
        # A = A * (1.0 - A)
        # X = tf.convert_to_tensor(X, dtype=tf.float32)
        return tf.reduce_mean(tf.nn.log_softmax(A))
    
    def update_params(self, lr):
        for i in range(1, self.L):
            self.W[i].assign_sub(lr * self.dW[i])
            self.b[i].assign_sub(lr * self.db[i])

    def predict(self, X):
        A = self.forward_pass(X)
        return tf.argmax(tf.nn.relu(A), axis=1)
    
    def info(self):
        num_params = 0
        for i in range(1, self.L):
            num_params += self.W[i].shape[0] * self.W[i].shape[1]
            num_params += self.b[i].shape[0]
        print('Input Features:', self.num_features)
        print('Number of Classes:', self.num_classes)
        print('Hidden Layers:')
        print('--------------')
        for i in range(1, self.L-1):
            print('Layer {}, Units {}'.format(i, self.layers[i]))
        print('--------------')
        print('Number of parameters:', num_params)

    def train_on_batch(self, X, Y, lr):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        Y = tf.convert_to_tensor(Y, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape:
            A = self.forward_pass(X)
            loss = self.compute_loss(A, Y)
        for key in self.W.keys():
            self.dW[key] = tape.gradient(loss, self.W[key])
            if self.dW[key] is None:
                self.dW[key] = self.W[key] * (1.0 - self.W[key]) * random.randint(0,1)

            self.db[key] = tape.gradient(loss, self.b[key])
            if self.db[key] is None:
                self.db[key] = self.b[key] * (1.0 - self.b[key]) * random.randint(0,1)

        del tape
        self.update_params(lr)
        
        return loss.numpy()

    def train(self, x_train, y_train, x_test, y_test, epochs, steps_per_epoch, batch_size, lr):
        history = {
            'val_loss': [],
            'train_loss': [],
            'val_acc': []
        }
        
        for e in range(0, epochs):
            epoch_train_loss = 0.
            print('Epoch {}'.format(e), end='.')
            for i in range(0, steps_per_epoch):
                x_batch = x_train[i*batch_size:(i+1)*batch_size]
                y_batch = y_train[i*batch_size:(i+1)*batch_size]
                batch_loss = self.train_on_batch(x_batch, y_batch, lr)
                epoch_train_loss += batch_loss
                
                if i%int(steps_per_epoch/10) == 0:
                    print(end='.')
                    
            history['train_loss'].append(epoch_train_loss/steps_per_epoch)

            val_A = self.forward_pass(x_test)
            history['val_loss'].append(self.compute_loss(val_A, y_test).numpy())
            
            
            val_preds = self.predict(x_test)
            history['val_acc'].append(np.mean(np.argmax(y_test, axis=1) == val_preds.numpy()))
            
            print('Val Acc:', history['val_acc'][-1], 'Validation Loss:', history['val_loss'][-1])


        
        return history
