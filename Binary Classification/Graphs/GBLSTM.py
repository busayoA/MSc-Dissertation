import tensorflow as tf
import numpy as np
from statistics import mean
from keras.models import Sequential
from keras.layers import Input, Bidirectional, Dense
from torch import lstm
from BinaryGraphInputLayer import BinaryGraphInputLayer as BGIL
from HiddenGraphLayer import HiddenGraphLayer as HGL

bgil = BGIL()
x_train_nodes, x_train_matrix, y_train, x_test_nodes, x_test_matrix, y_test = bgil.readFiles()
x_train = bgil.prepareData(x_train_nodes, x_train_matrix)


def GBLSTMModel(x_train, y_train, hiddenActivationFunction):
    index = 0
    accuracy = []
    for graph in x_train:
        graphs = [graph, graph]
        y = [y_train[index], y_train[index]]
        
        
        y = tf.convert_to_tensor(y)

        graphs = tf.convert_to_tensor(graphs, dtype=np.float32)
        dimensions = tf.shape(graphs[0], out_type=np.int32)

        
        lstmLayer1 = HGL(0.003, "lstm", hiddenActivationFunction, 64, 0.3, 2, [64, 64])
        lstmLayer1 = lstmLayer1.chooseModel((dimensions[0], ))
        # lstmLayer1 = LSTM(64, input_shape=(dimensions[0], dimensions[1]))
        lstmLayer2 = HGL(0.003, "lstm", hiddenActivationFunction, 64, 0.3, 2, [64, 64])
        lstmLayer2 = lstmLayer2.chooseModel((dimensions[0], 1))
        dropout = HGL(0.003, "dropout", "relu", 64, 0.3)
        output = HGL(0.003, "output", "relu", 2)
        model = Sequential()
        model.add(Input(shape=(dimensions[0], 1)))
        model.add(Bidirectional(lstmLayer1))
        model.add(dropout.chooseModel())
        model.add(output.chooseModel())
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()
        
        history = model.fit(graphs, y, epochs=10)

        accuracy.append(mean(history.history['accuracy']))
            
        index += 1
    print(hiddenActivationFunction, "completed")
    return sum(accuracy)/len(x_train)


withSigmoid = GBLSTMModel(x_train, y_train, "logSigmoid")
withTanh = GBLSTMModel(x_train, y_train, "tanh")
withSoftmax = GBLSTMModel(x_train, y_train, "softmax")
withRelu = GBLSTMModel(x_train, y_train, "relu")

print("With Sigmoid:", withSigmoid, "\nWith Tanh:", withTanh, "\nWith Softmax:", withSoftmax, "\nWith Relu:", withRelu)
