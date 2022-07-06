import tensorflow as tf
import numpy as np
from typing import List
from Node import Node
from AbstractTree import AbstractTree
from keras.models import Sequential

class TreeRNN(AbstractTree):
    def __init__(self, trees: List[Node], labels, layers, activationFunction: str, learningRate: float, 
    epochs: int):
       super().__init__(trees, labels, layers, activationFunction, learningRate, epochs)

    def initialiseWeights(self):
        # input Layer

        for i in range(1, self.layerCount):
            self.weights[i] = tf.Variable(tf.random.normal(shape=(self.layers[i],  self.layers[i-1])))
            self.bias[i] = tf.Variable(tf.random.normal(shape=(self.layers[i], 1)))

    def updateWeights(self):
        for i in range(1, self.layerCount):
            if self.weightDeltas[i] is not None:
                self.weights[i].assign_sub(self.learningRate * self.weightDeltas[i])

            if self.biasDeltas[i] is not None:
                self.bias[i].assign_sub(self.learningRate * self.biasDeltas[i])

    def FFLayer(self, tree):
        for i in range(1, self.layerCount): 
            tree = tf.matmul(tree, tf.transpose(self.weights[i])) + tf.transpose(self.bias[i])
            tree = self.activationFunction(tree)
        return tree

    def segmentationFunction(self, segmentationFunction: str):
        segmentationFunction = segmentationFunction.split("_")
        if segmentationFunction[0] == "sorted":
            if segmentationFunction[1] == "sum":
                return tf.math.segment_sum
            if segmentationFunction[1] == "mean":
                return tf.math.segment_mean
            if segmentationFunction[1] == "max":
                return tf.math.segment_max
            if segmentationFunction[1] == "min":
                return tf.math.segment_min
            if segmentationFunction[1] == "prod":
                return tf.math.segment_prod
        elif segmentationFunction[0] == "unsorted":
            if segmentationFunction[1] == "sum":
                return tf.math.unsorted_segment_sum
            if segmentationFunction[1] == "mean":
                return tf.math.unsorted_segment_mean
            if segmentationFunction[1] == "max":
                return tf.math.unsorted_segment_max
            if segmentationFunction[1] == "min":
                return tf.math.unsorted_segment_min
            if segmentationFunction[1] == "prod":
                return tf.math.unsorted_segment_prod
        else:
            return None

    def segmentationLayer(self, segmentationFunction: str, nodeEmbeddings: tf.Tensor):
        seg = segmentationFunction.lower()
        segmentationFunction = self.segmentationFunction(segmentationFunction)

        if seg.split("_")[0] == "unsorted":
            return segmentationFunction(nodeEmbeddings, tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 10)
        else:
            return segmentationFunction(nodeEmbeddings, tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

    def backPropagate(self, tree, yValues):
        with tf.GradientTape(persistent=True) as tape: 
            output = self.FFLayer(tree)
            loss = self.lossFunction(output, yValues)
        
        for i in range(1, self.layerCount):
            tape.watch(self.weights[i])
            self.weightDeltas[i] = tape.gradient(loss, self.weights[i])
            self.biasDeltas[i] = tape.gradient(loss, self.bias[i])
        
        del tape
        self.updateWeights()
        return loss.numpy()

    def runFFModel(self, x_train, y_train, x_test, y_test):
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

            # FIRST FORWARD PASS
            loss = self.backPropagate(x_train, y_train[index])
            # SECOND FORWARD PASS/RECURRENT LOOP
            newOutputs = self.FFLayer(x_train)
            pred = tf.argmax(tf.nn.softmax(newOutputs), axis = 1)
            predictions.append(pred)
            index += 1

            predictions = tf.convert_to_tensor(predictions)
            # print(predictions)
            metrics['trainingLoss'].append(loss)
            unseenPredictions = self.makePrediction(x_test)
            metrics['trainingAccuracy'].append(np.mean(np.argmax(y_train, axis=1) == predictions.numpy()))
            metrics['validationAccuracy'].append(np.mean(np.argmax(y_test, axis=1) == unseenPredictions.numpy()))
            print('\tLoss:', metrics['trainingLoss'][-1], 'Accuracy:', metrics['trainingAccuracy'][-1], 
            'Validation Accuracy:', metrics['validationAccuracy'][-1])
        return metrics

    def runRNNModel(self, x_train, y_train, x_test, y_test, filename):
        model = Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(x_train.shape[1], 1)))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, input_shape=(x_train.shape[1], 1), return_sequences=True)))
        model.add(tf.keras.layers.LSTM(128))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(model.summary())
        
        model.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_test, y_test))
        
        model.save(filename)
