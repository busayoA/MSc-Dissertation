from TreeInputLayer import TreeInputLayer
from HiddenTreeLayer import HiddenTreeLayer

input = TreeInputLayer()
x_train_all, y_train, x_test_all, y_test = input.splitTrainTest(False)

x_train_trees, x_train_nodes = [], []
for i in range(len(x_train_all)):
    x_train_nodes.append(x_train_all[i][0])
    x_train_trees.append(x_train_all[i][1])

x_test_trees, x_test_nodes = [], []
for i in range(len(x_test_all)):
    x_test_nodes.append(x_test_all[i][0])
    x_test_trees.append(x_test_all[i][1])


# USING RELU ACTIVATION
print("WITH RELU")
hidden1 = HiddenTreeLayer(x_train_trees, y_train, [len(x_train_nodes[0]), len(x_train_nodes[0]), len(x_train_nodes[0]), 2], "relu", 0.03, 5, "max")
x_train = hidden1.prepareAllEmbeddings(x_train_nodes)
x_test = hidden1.prepareAllEmbeddings(x_test_nodes)

hidden1.runModelWithBackPropagation(x_train, y_train, x_test, y_test)
hidden1.runModel(x_train, y_train, x_test, y_test)


# USING SOFTMAX ACTIVATION
print("WITH SOFTMAX")
hidden2 = HiddenTreeLayer(x_train_trees, y_train, [len(x_train_nodes[0]), len(x_train_nodes[0]), len(x_train_nodes[0]), 2], "softmax", 0.03, 5, "max")
x_train = hidden2.prepareAllEmbeddings(x_train_nodes)
x_test = hidden2.prepareAllEmbeddings(x_test_nodes)

hidden2.runModelWithBackPropagation(x_train, y_train, x_test, y_test)
hidden2.runModel(x_train, y_train, x_test, y_test)

# USING TANH ACTIVATION
print("WITH TANH")
hidden3 = HiddenTreeLayer(x_train_trees, y_train, [len(x_train_nodes[0]), len(x_train_nodes[0]), len(x_train_nodes[0]), 2], "tanh", 0.03, 5, "max")
x_train = hidden3.prepareAllEmbeddings(x_train_nodes)
x_test = hidden3.prepareAllEmbeddings(x_test_nodes)

hidden3.runModelWithBackPropagation(x_train, y_train, x_test, y_test)
hidden3.runModel(x_train, y_train, x_test, y_test)


# USING LOGARITHMIC SIGMOID ACTIVATION
print("WITH LOGARITHMIC SIGMOID")
hidden4 = HiddenTreeLayer(x_train_trees, y_train, [len(x_train_nodes[0]), len(x_train_nodes[0]), len(x_train_nodes[0]), 2], "logsigmoid", 0.03, 5, "max")
x_train = hidden4.prepareAllEmbeddings(x_train_nodes)
x_test = hidden4.prepareAllEmbeddings(x_test_nodes)

hidden4.runModelWithBackPropagation(x_train, y_train, x_test, y_test)
hidden4.runModel(x_train, y_train, x_test, y_test)
