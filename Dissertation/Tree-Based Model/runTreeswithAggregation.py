import DataProcessor as dp
import tensorflow as tf
from TreeRNN import TreeRNN
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D

""" RUN THE UNHASHED MODEL """
x_train, y_train, x_test, y_test = dp.getData(False)
# print(x_train)
seg = TreeRNN(x_train, y_train, [0], "", 0.0, 0)

x_train_pooled = []
x_test_pooled = []




""" EXPERIMENTS WITH ALL TYPES OF AGGREGATION"""

# USING RELU ACTIVATION
# print("UNHASHED WITH RELU")
# x_train_means = tf.reshape(x_train_means, (1, (len(x_train_means))))
# hidden1 = TreeRNN(x_train_means, y_train, [tf.shape(x_train_means)[1], 64, 2], "relu", 0.03, 20)
# hidden1.runModel(x_train_means, y_train, x_test_means, y_test)

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

