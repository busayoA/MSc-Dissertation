import tensorflow as tf
import numpy as np
from BinaryGraphInputLayer import BinaryGraphInputLayer as BGIL
from HiddenGraphLayer import HiddenGraphLayer as HGL
from os.path import dirname, join
current_dir = dirname(__file__)


merge = "./Data/Merge Sort"
quick = "./Data/Quick Sort"

merge = join(current_dir, merge)
quick = join(current_dir, quick)
# merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort"
# quick = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Quick Sort"

inputLayer = BGIL()
xTrain, yTrain, xTest, yTest = inputLayer.splitTrainTest(merge, quick)
x_train, x_train_matrix, y_train, x_test, x_test_matrix, y_test = inputLayer.getDatasets(xTrain, yTrain, xTest, yTest)

def getAdjacencyLists(x_list, x_matrix, embeddingFileToWrite, adjFileToWrite):
    print("Collecting node embeddings and adjacency lists")
    adj = [0] * len(x_list)
    for i in range(len(x_list)):
        x_list[i], adj[i] = inputLayer.getAdjacencyLists(x_list[i], x_matrix[i])

    return x_list, adj
    # with open(embeddingFileToWrite, 'w') as writer:
    #     for i in range(len(x_list)):
    #         x = str(x_list[i]) + "\n"
    #         writer.write(x)

    # with open(adjFileToWrite, "w") as writer:
    #     for i in range(len(adj)):
    #         x = str(adj[i])
    #         writer.write(x)


x_train, x_train_adj = getAdjacencyLists(x_train, x_train_matrix, "x_train_embeddings.txt", "x_train_adj.txt")
x_test, x_test_adj = getAdjacencyLists(x_test, x_test_matrix, "x_test_embeddings.txt", "x_test_adj.txt")
print(x_train)

for i in range(len(x_train)):
    x_train[i] = tf.convert_to_tensor(x_train[i], dtype=np.float32)
    # x_train_adj[i] = tf.convert_to_tensor(x_train_adj[i][1], dtype=np.float32)

for i in range(len(x_test)):
    x_test[i] = tf.convert_to_tensor(x_test[i], dtype=np.float32)
    
print(x_train_adj[0])
# file_path = join(current_dir, "./x_train_embeddings.txt")
# with open(file_path, 'r') as f:
#     x_train = f.read()


# x_train = x_train.split("]")
# for i in range(len(x_train)):
#     x_train[i] = x_train[i].replace("[", "").replace(" ", "")
#     x_train[i] = x_train[i].split(",")
#     for j in range(len(x_train[i])):
#         x_train[i][j] = float(x_train[i][j])
#     print(x_train[i])

# x_train =tf.convert_to_tensor(x_train, dtype=np.float32)

# print(x_train[0])


# learningRate = 0.03
# layerName = "feedforward"
# activationFunction = "logSigmoid"
# neurons = 64
# dropoutRate = 0.03
# hiddenLayers = [neurons, neurons]
# hgl = HGL(learningRate, layerName, activationFunction, neurons, dropoutRate, len(hiddenLayers), hiddenLayers)
# ffModel = hgl.chooseModel()
# print(ffModel)
