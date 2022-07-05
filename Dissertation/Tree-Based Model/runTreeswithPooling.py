import DataProcessor as dp
from TreeRNN import TreeRNN
import tensorflow as tf



x_train, y_train, x_test, y_test = dp.getData(False)
seg = TreeRNN(x_train, y_train, [0], "", 0.0, 0)
x_train_pooled, x_test_pooled = [], []
for i in x_train:
    i = tf.convert_to_tensor(i)
    shape = i.shape[0]
    i = tf.reshape(i, (1, i.shape[0], 1))
    globalPool1d = tf.keras.layers.GlobalAveragePooling1D()(i)

    globalPool1d = globalPool1d.numpy()
    # globalPool1d = globalPool1d.swapaxes(1).reshape(shape,-1)

    x_train_pooled.append(globalPool1d[0][0])

for i in x_test:
    i = tf.convert_to_tensor(i)
    shape = i.shape[0]
    i = tf.reshape(i, (1, i.shape[0], 1))
    globalPool1d = tf.keras.layers.GlobalAveragePooling1D()(i)

    globalPool1d = globalPool1d.numpy()
    # globalPool1d = globalPool1d.swapaxes(1).reshape(shape,-1)

    x_test_pooled.append(globalPool1d[0])


# x_train_pooled = tf.convert_to_tensor(x_train_pooled)
# print(x_train_pooled)
# """ RUN THE UNHASHED MODEL """

# # USING RELU ACTIVATION
print("UNHASHED WITH RELU")
hidden1 = TreeRNN(x_train_pooled, y_train, [1, 64, 64, 2], "relu", 0.03, 20)
hidden1.runModel(x_train_pooled, y_train, x_test, y_test)

# # USING SOFTMAX ACTIVATION
# print("UNHASHED WITH SOFTMAX")
# hidden2 = TreeRNN(x_train, y_train, [1, 64, 2], "softmax", 0.03, 20)
# hidden2.runModel(x_train, y_train, x_test, y_test)

# # USING TANH ACTIVATION
# print("UNHASHED WITH TANH")
# hidden3 = TreeRNN(x_train, y_train, [1, 64, 2], "tanh", 0.03, 20)
# hidden3.runModel(x_train, y_train, x_test, y_test)

# # USING LOGARITHMIC SIGMOID ACTIVATION
# print("UNHASHED WITH LOGARITHMIC SIGMOID")
# hidden4 = TreeRNN(x_train, y_train, [1, 64, 2], "logsigmoid", 0.03, 20)
# hidden4.runModel(x_train, y_train, x_test, y_test)


# """ RUN THE HASHED MODEL """
# x_train, y_train, x_test, y_test = dp.getData(True)
# # USING RELU ACTIVATION
# print("HASHED WITH RELU")
# hidden5 = TreeRNN(x_train, y_train, [1, 64, 2], "relu", 0.03, 20)
# hidden5.runModel(x_train, y_train, x_test, y_test)

# # USING SOFTMAX ACTIVATION
# print("HASHED WITH SOFTMAX")
# hidden6 = TreeRNN(x_train, y_train, [1, 64, 2], "softmax", 0.03, 20)
# hidden6.runModel(x_train, y_train, x_test, y_test)

# # USING TANH ACTIVATION
# print("HASHED WITH TANH")
# hidden7 = TreeRNN(x_train, y_train, [1, 64, 2], "tanh", 0.03, 20)
# hidden7.runModel(x_train, y_train, x_test, y_test)

# # USING LOGARITHMIC SIGMOID ACTIVATION
# print("HASHED WITH LOGARITHMIC SIGMOID")
# hidden8 = TreeRNN(x_train, y_train, [1, 64, 2], "logsigmoid", 0.03, 20)
# hidden8.runModel(x_train, y_train, x_test, y_test)

