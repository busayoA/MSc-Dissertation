import tensorflow as tf
import numpy as np
from statistics import mean
from keras.models import Sequential
from keras.layers import Input
from BinaryGraphInputLayer import BinaryGraphInputLayer as BGIL
from HiddenGraphLayer import HiddenGraphLayer as HGL

bgil = BGIL()
x_train_nodes, x_train_matrix, y_train, x_test_nodes, x_test_matrix, y_test = bgil.readFiles()
x_train = bgil.prepareData(x_train_nodes, x_train_matrix)

def GBRNNModel(x_train, y_train, hiddenActivationFunction):
    index = 0
    accuracy = []
    for graph in x_train:
        graphs = [graph, graph]
        y = [y_train[index], y_train[index]]
        y = tf.convert_to_tensor(y)

        graphs = tf.convert_to_tensor(graphs, dtype=np.float32)
        dimensions = tf.shape(graphs[0], out_type=np.int32)
        
        rnnLayer1 = HGL(0.003, "rnn", hiddenActivationFunction, 64, 0.3, 2, [64, 64])
        rnnLayer1 = rnnLayer1.chooseModel()
        output = HGL(0.003, "output", "relu", 2)
        model = Sequential()
        model.add(Input(shape=(dimensions[0], 1)))
        model.add(rnnLayer1)
        model.add(output.chooseModel())
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()
        
        history = model.fit(graphs, y, epochs=10)

        accuracy.append(mean(history.history['accuracy']))
            
        index += 1
    print(hiddenActivationFunction, "completed")
    return sum(accuracy)/len(x_train)


withSigmoid = GBRNNModel(x_train, y_train, "logSigmoid")
withTanh = GBRNNModel(x_train, y_train, "tanh")
withSoftmax = GBRNNModel(x_train, y_train, "softmax")
withRelu = GBRNNModel(x_train, y_train, "relu")

print("With Sigmoid:", withSigmoid, "\nWith Tanh:", withTanh, "\nWith Softmax:", withSoftmax, "\nWith Relu:", withRelu)
