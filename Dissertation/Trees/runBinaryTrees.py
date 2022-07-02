from TreeInputLayer import TreeInputLayer as TIL
from HiddenTreeLayer import HiddenTreeLayer as HTL

input = TIL()
x_train, y_train = input.getData(False)


# USING RELU ACTIVATION
print("WITH RELU")
hidden1 = HTL(x_train, y_train, [1, 158, 128, 2], "relu", 0.03, 5)
outputs = hidden1.prepareEmbeddings()

hidden1.runModel(outputs, y_train, "max")
hidden1.runModelWithBackPropagation(outputs, y_train, "max")


# USING SOFTMAX ACTIVATION
print("WITH SOFTMAX")
hidden2 = HTL(x_train, y_train, [1, 158, 128, 2], "softmax", 0.03, 5)
outputs = hidden2.prepareEmbeddings()

hidden2.runModel(outputs, y_train, "max")
hidden2.runModelWithBackPropagation(outputs, y_train, "max")


# USING TANH ACTIVATION
print("WITH TANH")
hidden3 = HTL(x_train, y_train, [1, 158, 128, 2], "tanh", 0.03, 5)
outputs = hidden3.prepareEmbeddings()

hidden3.runModel(outputs, y_train, "max")
hidden3.runModelWithBackPropagation(outputs, y_train, "max")


# USING LOGARITHMIC SIGMOID ACTIVATION
print("WITH LOGARITHMIC SIGMOID")
hidden4 = HTL(x_train, y_train, [1, 158, 128, 2], "logsigmoid", 0.03, 5)
outputs = hidden4.prepareEmbeddings()

hidden4.runModel(outputs, y_train, "max")
hidden4.runModelWithBackPropagation(outputs, y_train, "max")
