import numpy as np
from RNN import RNN
from GraphDataProcessor import GraphDataProcessor

gdp = GraphDataProcessor()
"""RUNNING ON PADDED GRAPHS"""
x_train, y_train, x_test, y_test = gdp.runProcessor1()

layers = [len(x_train[0]), 128, 128, 2]


graphLSTM = "lstm"
gru = "gru"
simpleRNN = "rnn"

rnn = RNN("lstm", x_train, y_train, x_test, y_test, "relu")

graphLSTM = rnn.runModel(graphLSTM, "graphLSTM", 256, 30)





# """RUNNING ON SEGMENTED GRAPHS"""
# x_train, y_train, x_test, y_test = gdp.runProcessor3()

# layers = [len(x_train[0]), 128, 128, 2]
# epochs = 10
# lr = 0.001
# mlp1 = MLP(x_train, y_train, layers, "relu", lr, epochs)
# metrics1 = mlp1.runFFModel(x_train, y_train, x_test, y_test)

# mlp2 = MLP(x_train, y_train, layers, "tanh", lr, epochs)
# metrics2 = mlp2.runFFModel(x_train, y_train, x_test, y_test)

# mlp3 = MLP(x_train, y_train, layers, "softmax", lr, epochs)
# metrics3 = mlp3.runFFModel(x_train, y_train, x_test, y_test)

# mlp4 = MLP(x_train, y_train, layers, "logsigmoid", lr, epochs)
# metrics4 = mlp4.runFFModel(x_train, y_train, x_test, y_test)


# print("USING THE MULTI-LAYER PERCEPTRON")
# print("USING RELU")
# print("Average loss:", np.average(metrics1['trainingLoss']), "Average training accuracy:", 
# np.average(metrics1['trainingAccuracy']), "Average validation accuracy:", 
# np.average(metrics1['validationAccuracy']), "\n") 

# print("USING TANH")
# print("Average loss:", np.average(metrics2['trainingLoss']), "Average training accuracy:", 
# np.average(metrics2['trainingAccuracy']), "Average validation accuracy:", 
# np.average(metrics2['validationAccuracy']), "\n") 

# print("USING SOFTMAX")
# print("Average loss:", np.average(metrics3['trainingLoss']), "Average training accuracy:", 
# np.average(metrics3['trainingAccuracy']), "Average validation accuracy:", 
# np.average(metrics3['validationAccuracy']), "\n") 

# print("USING SIGMOID")
# print("Average loss:", np.average(metrics4['trainingLoss']), "Average training accuracy:", 
# np.average(metrics4['trainingAccuracy']), "Average validation accuracy:", 
# np.average(metrics4['validationAccuracy']), "\n") 