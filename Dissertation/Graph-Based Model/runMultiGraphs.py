import tensorflow as tf
import numpy as np
from statistics import mean
from keras.models import Sequential
from keras.layers import Input, Bidirectional
from GraphInputLayer import GraphInputLayer as GIL
from HiddenGraphLayer import HiddenGraphLayer as HGL

input = GIL()
x_train_nodes, x_train_matrix, y_train = input.readFiles()
x_train = input.prepareData(x_train_nodes, x_train_matrix)


def MGBFFNModel(x_train, y_train, hiddenActivationFunction):
    index = 0
    accuracy = []
    for graph in x_train:
        graphs = [graph, graph]
        y = [y_train[index], y_train[index]]

        dimensions = tf.shape(graphs[0], out_type=np.int32)
        ffLayer = HGL(0.003, "ffn", hiddenActivationFunction, 64, 0.3, 2, [64, 64])
        output = HGL(0.003, "output", "relu", 3)
        model = Sequential()
        model.add(Input(shape=(dimensions[0], )))
        for i in range(2):
            model.add(ffLayer.chooseModel()[i])
        model.add(output.chooseModel())
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # model.summary()
        y = tf.convert_to_tensor(y)

        # graphs = np.reshape(graphs, (2, dimensions[1]))
        graphs = tf.convert_to_tensor(graphs, dtype=np.float32)
        history = model.fit(graphs, y, epochs=10)

        accuracy.append(mean(history.history['accuracy']))
            
        index += 1
    print(hiddenActivationFunction, "completed")
    return sum(accuracy)/len(x_train)


withSigmoid = MGBFFNModel(x_train, y_train, "logSigmoid")
withTanh = MGBFFNModel(x_train, y_train, "tanh")
withSoftmax = MGBFFNModel(x_train, y_train, "softmax")
withRelu = MGBFFNModel(x_train, y_train, "relu")

print("With Sigmoid:", withSigmoid, "\nWith Tanh:", withTanh, "\nWith Softmax:", withSoftmax, "\nWith Relu:", withRelu)



def MGBRNNModel(x_train, y_train, hiddenActivationFunction):
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
        output = HGL(0.003, "output", "relu", 3)
        model = Sequential()
        model.add(Input(shape=(dimensions[0], 1)))
        model.add(rnnLayer1)
        model.add(output.chooseModel())
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()
        
        history = model.fit(graphs, y, epochs=10)

        accuracy.append(mean(history.history['accuracy']))
            
        index += 1
    print(hiddenActivationFunction, "completed")
    return sum(accuracy)/len(x_train)


withSigmoid = MGBRNNModel(x_train, y_train, "logSigmoid")
withTanh = MGBRNNModel(x_train, y_train, "tanh")
withSoftmax = MGBRNNModel(x_train, y_train, "softmax")
withRelu = MGBRNNModel(x_train, y_train, "relu")

print("With Sigmoid:", withSigmoid, "\nWith Tanh:", withTanh, "\nWith Softmax:", withSoftmax, "\nWith Relu:", withRelu)



def MGBLSTMModel(x_train, y_train, hiddenActivationFunction):
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
        output = HGL(0.003, "output", "relu", 3)
        model = Sequential()
        model.add(Input(shape=(dimensions[0], 1)))
        model.add(Bidirectional(lstmLayer1))
        model.add(dropout.chooseModel())
        model.add(output.chooseModel())
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()
        
        history = model.fit(graphs, y, epochs=10)

        accuracy.append(mean(history.history['accuracy']))
            
        index += 1
    print(hiddenActivationFunction, "completed")
    return sum(accuracy)/len(x_train)


withSigmoid = MGBLSTMModel(x_train, y_train, "logSigmoid")
withTanh = MGBLSTMModel(x_train, y_train, "tanh")
withSoftmax = MGBLSTMModel(x_train, y_train, "softmax")
withRelu = MGBLSTMModel(x_train, y_train, "relu")

print("With Sigmoid:", withSigmoid, "\nWith Tanh:", withTanh, "\nWith Softmax:", withSoftmax, "\nWith Relu:", withRelu)

