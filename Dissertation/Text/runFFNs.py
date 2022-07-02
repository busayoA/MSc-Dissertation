import readFiles as rf
import tensorflow as tf
import numpy as np
from SimpleFFN import FeedForwardNetwork


# BINARY MODEL
x_train, y_train, x_test, y_test = rf.getVectorizedCodeData(False)

epochs = 10
lr = 0.001
rnn = FeedForwardNetwork([len(x_train[0]), 128, 128, 2], epochs, lr)

# print(rnn.backPropagate(x_train, y_train))
# print(rnn.predict(x_test, y_test))
metrics = rnn.trainModel(x_train, y_train, x_test, y_test)
print("Average loss:", np.average(metrics['trainingLoss']), "Average accuracy:", np.average(metrics['accuracy']))


# MULTI-CLASS MODEL
x_train, y_train, x_test, y_test = rf.getVectorizedCodeData(True)

epochs = 10
lr = 0.001
rnn = FeedForwardNetwork([len(x_train[0]), 128, 128, 3], epochs, lr)

# print(rnn.backPropagate(x_train, y_train))
# print(rnn.predict(x_test, y_test))
metrics = rnn.trainModel(x_train, y_train, x_test, y_test)
print("Average loss:", np.average(metrics['trainingLoss']), "Average accuracy:", np.average(metrics['accuracy']))


