import tensorflow as tf
import numpy as np
from statistics import mean
from keras.models import Sequential
from keras.layers import Input
from BinaryGraphInputLayer import BinaryGraphInputLayer as BGIL
from HiddenGraphLayer import HiddenGraphLayer as HGL
from os.path import dirname, join
current_dir = dirname(__file__)


merge = "./Data/Merge Sort"
quick = "./Data/Quick Sort"

merge = join(current_dir, merge)
quick = join(current_dir, quick)

inputLayer = BGIL()
xTrain, yTrain, xTest, yTest = inputLayer.splitTrainTest(merge, quick)
x_train_nodes, x_train_matrix, y_train, x_test_nodes, x_test_matrix, y_test = inputLayer.getDatasets(xTrain, yTrain, xTest, yTest)

def getAdjacencyLists(x_list, x_matrix):
    print("Collecting node embeddings and adjacency lists")
    adj = [0] * len(x_list)
    for i in range(len(x_list)):
        x_list[i], adj[i] = inputLayer.getAdjacencyLists(x_list[i], x_matrix[i])

    return x_list, adj

x_train, x_train_adj = getAdjacencyLists(x_train_nodes, x_train_matrix)
x_test, x_test_adj = getAdjacencyLists(x_test_nodes, x_test_matrix)
# print(x_train)

for i in range(len(x_train)):
    x_train[i] = tf.convert_to_tensor(x_train[i], dtype=np.float32)

for i in range(len(x_test)):
    x_test[i] = tf.convert_to_tensor(x_test[i], dtype=np.float32)
    
# for value in x_train_adj:
#     for node in value:
#         print(value[node])



index = 0
x = []
for graph in x_train:
    graphs = [graph, graph]
    y = [y_train[index], y_train[index]]

    adjacencies = x_train_adj[index]
    originalNodes = x_train_nodes[index]
    dimensions = tf.shape(graphs, out_type=np.int32)
    
    ffLayer = HGL(0.003, "feedforward", "logSigmoid", 64, 0.3, 2, [64, 64])
    output = HGL(0.003, "output", "relu", 2)
    model = Sequential()
    model.add(Input(shape=(dimensions[1], )))
    for i in range(2):
        model.add(ffLayer.chooseModel()[i])
    model.add(output.chooseModel())
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
   
    model.summary()
    y = tf.convert_to_tensor(y)

    graphs = np.reshape(graphs, (2, dimensions[1]))
    graphs = tf.convert_to_tensor(graphs, dtype=np.float32)
    history = model.fit(graphs, y, epochs=10)
    # print(ffModel)
    x.append(mean(history.history['accuracy']))
    
    index += 1
    
print(sum(x)/96)
    # for i in range(len(adjacencies)):
    #     for j in range(len(graph)):
    #         if x_train_nodes[index][j] == adjacencies[i][1]:
    #             overallModel = tf.keras.models.Sequential() 
    #             neuronUnits = len(adjacencies[i][2])
    #             newHGL = HGL(learningRate, layerName, activationFunction, len(adjacencies[i][2]), dropoutRate, 1, [len(adjacencies[i][2])])
    #             model = newHGL.chooseModel()
    #             overallModel.add(model[0])
    #             overallModel.compile()
    #             overallModel.build(input_shape = len(adjacencies[i][2]))
    #             overallModel.summary()
