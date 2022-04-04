from turtle import xcor
from cv2 import mean
import NeuralNetworks.RNN as RNN
import RNN3 as RNN3
import preTraining as pt, readFiles as rf, tensorflow as tf
import numpy as np
import readFiles as rf
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def createEmbeddings(xData):
    embeddings = CountVectorizer(tokenizer=lambda doc: doc, min_df=2).fit_transform(xData)
    embeddings = embeddings.toarray()
    return embeddings

x_train, y_train, x_test, y_test = rf.createTrainTestData()

x_train = createEmbeddings(x_train)
x_test = createEmbeddings(x_test)
x_train = np.reshape(x_train, (x_train.shape[0], len(x_train[0])))/255.
x_test =  np.reshape(x_test, (x_test.shape[0], len(x_test[0])))/255.
# y_train = tf.keras.utils.to_categorical(y_train)
# y_test = tf.keras.utils.to_categorical(y_test)

# rnn  = RNN.RNN([len(x_train), 20, 20, len(y_train)])


# batch_size = 50
# epochs = 10
# steps_per_epoch = int(x_train.shape[0]/5)
# lr = 0.4

# print('Steps per epoch:', steps_per_epoch)

# history = rnn.train(
#     x_train, y_train,
#     x_test, y_test,
#     epochs, steps_per_epoch,
#     batch_size, lr
# )

batchSize = 50
epochs = 10
lr = 0.2
stepsPerEpoch = int(x_train.shape[0]/5)

row1 = x_train[0]
rnn  = RNN3.RNN([len(x_train), 20, 20, 5], epochs, lr, batchSize, stepsPerEpoch)
# rnn.forwardPass(row1)
rnn.trainModel(x_train, y_train, x_test, y_test)
# print(rnn.predict(x_train))