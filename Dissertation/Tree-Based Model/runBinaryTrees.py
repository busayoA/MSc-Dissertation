import TreeEmbeddingLayer as embeddingLayer
from UnpaddedTreeRNN import UnpaddedTreeRNN
from PaddedTreeRNN import PaddedTreeRNN

x_train, y_train, x_test, y_test = embeddingLayer.getData(False)

""" RUN THE UNPADDED MODEL"""
# USING RELU ACTIVATION
print("WITH RELU")
hidden1 = UnpaddedTreeRNN(x_train, y_train, [1, 64, 2], "relu", 0.03, 5)
hidden1.runModel(x_train, y_train, x_test, y_test)

# USING SOFTMAX ACTIVATION
print("WITH SOFTMAX")
hidden2 = UnpaddedTreeRNN(x_train, y_train, [1, 64, 2], "softmax", 0.03, 5)
hidden2.runModel(x_train, y_train, x_test, y_test)

# USING TANH ACTIVATION
print("WITH TANH")
hidden3 = UnpaddedTreeRNN(x_train, y_train, [1, 64, 2], "tanh", 0.03, 5)
hidden3.runModel(x_train, y_train, x_test, y_test)

# USING LOGARITHMIC SIGMOID ACTIVATION
print("WITH LOGARITHMIC SIGMOID")
hidden4 = UnpaddedTreeRNN(x_train, y_train, [1, 64, 2], "logsigmoid", 0.03, 5)
hidden4.runModel(x_train, y_train, x_test, y_test)
