import tensorflow as tf
import numpy as np
from typing import List

class MLP:
    def __init__(self, x_train: tf.Tensor, y_train: List, layers: List[int],
                 activationFunction: str, learningRate: float, epochs: int):
        """ The Multi Layer Perceptron Class  
        x_train: tf.Tensor - The training data 
        y_train: List - The training data labels
        layers: List[int] - The list of layer units
        activationFunction: str - The activation function to be used
        learningRate: float - The learning rate
        epochs: int - The number of times the data is passed back and forth in the network"""
        self.x_train = x_train
        self.y_train = y_train
        self.layers = layers
        self.layerCount = len(self.layers)
        self.xCount = len(self.x_train)
        self.activationFunction = self.getActivationFunction(activationFunction)
        self.learningRate = learningRate
        self.epochs = epochs
        self.weights, self.bias, self.weightDeltas, self.biasDeltas = {}, {}, {}, {}
        self.hiddenLayerCount = len(layers)-2
        self.featureCount = layers[0]
        self.classCount = layers[-1]
        self.parameterCount = 0

        self.initialiseWeights()

        # Calculate the number of parameters in the model
        for i in range(1, self.layerCount):
            self.parameterCount += self.weights[i].shape[0] * self.weights[i].shape[1]
            self.parameterCount += self.bias[i].shape[0]

        # Print a summary of the model
        print(self.featureCount, "features,", self.classCount, "classes,", self.parameterCount, "parameters, and", self.hiddenLayerCount, "hidden layers", "\n")
        for i in range(1, self.layerCount-1):
            print('Hidden layer {}:'.format(i), '{} neurons'.format(self.layers[i]))

    def getActivationFunction(self, activationFunction: str):
        """ The method to seelct an activation function
        activationFunction: str - The string representation of the activation function"""
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

    def lossFunction(self, outputs: tf.Tensor, yValues: tf.Tensor):
        """ Calculate the loss between the actual and predicted outputs """
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yValues, outputs))

    def makePrediction(self, x_test: tf.Tensor):
        """Make predictions on the unseen testing data
        x_test: tf.Tensor - The unseen testing data in tensor format"""
        predictions = []
        output = self.FFLayer(x_test)
        prediction = tf.argmax(tf.nn.softmax(output), axis=1)
        predictions.append(prediction)
        return tf.convert_to_tensor(predictions)

    def initialiseWeights(self):
        """Initialise the weights and biases for the model"""
        for i in range(1, self.layerCount):
            self.weights[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], self.layers[i-1])))
            self.bias[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], 1)))

    def FFLayer(self, x: tf.Tensor):
        """ The Feed Forward layer for the model
        x: tf.Tensor - The data to be passed across the layers """
        for i in range(1, self.layerCount): 
            x = tf.matmul(x, tf.transpose(self.weights[i])) + tf.transpose(self.bias[i])
            x = self.activationFunction(x)
        return x

    def backPropagate(self, xValues: tf.Tensor, yValues: tf.Tensor):
        """ Run the backpropagation algorithm on the training data 
        xValues: tf.Tensor - The training data
        yValues: tf.Tensor - The training data labels
        """
        with tf.GradientTape(persistent=True) as tape: 
            output = self.FFLayer(xValues)
            loss = self.lossFunction(output, yValues)
        
        for i in range(1, self.layerCount):
            tape.watch(self.weights[i])
            # calcualte the gradients of the weights and biases
            self.weightDeltas[i] = tape.gradient(loss, self.weights[i])
            self.biasDeltas[i] = tape.gradient(loss, self.bias[i])
        
        del tape
        # update the weights after calculating the gradients
        self.updateWeights()
        return loss.numpy()

    def updateWeights(self):
        """The method that updates the weights and biases after backpropagation 
        has been carried out"""
        for i in range(1, self.layerCount):
            if self.weightDeltas[i] is not None:
                # Subtract the learning rate * the gradient from the current weight/bias
                self.weights[i].assign_sub(self.learningRate * self.weightDeltas[i])

            if self.biasDeltas[i] is not None:
                self.bias[i].assign_sub(self.learningRate * self.biasDeltas[i])

    def runFFModel(self, x_train: tf.Tensor, y_train: tf.Tensor, x_test: tf.Tensor, y_test: tf.Tensor):
        """ The method to implement the MLP
        x_train: tf.Tensor - The training data
        y_train: tf.Tensor - The training data labels
        x_test: tf.Tensor - The testing data
        y_test: tf.Tensor - The testing data labels"""
        index = 0
        loss = 0.0
        metrics = {'trainingLoss': [], 'trainingAccuracy': [], 'validationAccuracy': []}
        self.initialiseWeights()
        for i in range(self.epochs):
            print('Epoch {}'.format(i), end='........')
            predictions = []

            if index % 5 == 0:
                print(end=".")
            if index >= len(y_train):
                index = 0

            # First forward pass
            loss = self.backPropagate(x_train, y_train)
            # Second forward pass/Recurrent Loop with the updated weights
            newOutputs = self.FFLayer(x_train)

            # run softmax activation on the outputs to get the final prediction
            pred = tf.argmax(tf.nn.softmax(newOutputs), axis = 1)
            predictions.append(pred)
            index += 1

            predictions = tf.convert_to_tensor(predictions)

            # add the calculated loss to the list of metrics
            metrics['trainingLoss'].append(loss)

            # make predictions on the unseen testing data
            unseenPredictions = self.makePrediction(x_test)

            # Calculate the accuracies of the seen and unseen predictions
            metrics['trainingAccuracy'].append(np.mean(np.argmax(y_train, axis=1) == predictions.numpy()))
            metrics['validationAccuracy'].append(np.mean(np.argmax(y_test, axis=1) == unseenPredictions.numpy()))
            print('\tLoss:', metrics['trainingLoss'][-1], 'Accuracy:', metrics['trainingAccuracy'][-1],
                  'Validation Accuracy:', metrics['validationAccuracy'][-1])
        return metrics

