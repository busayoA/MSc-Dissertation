import parseTextFiles as ptf
import numpy as np
from SimpleFFN import TextBasedModel

x_train, y_train, x_test, y_test = ptf.getVectorizedCodeData()

epochs = 10
lr = 0.001
rnn = TextBasedModel([len(x_train[0]), 128, 128, 2], epochs, lr)
metrics = rnn.runFFModel(x_train, y_train, x_test, y_test)
print("Average loss:", np.average(metrics['trainingLoss']), "Average accuracy:", np.average(metrics['trainingAccuracy']))


lstmModel = rnn.runRNNModel(x_train, y_train, x_test, y_test)

denseModel = rnn.runDenseModel(x_train, y_train, x_test, y_test)
