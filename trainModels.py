from cv2 import mean
import NeuralNetworks.RNN as RNN
import preTraining as pt, readFiles as rf, tensorflow as tf
import numpy as np
import readFiles as rf
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# vectorizer = CountVectorizer(min_df=20)

def createEmbeddings(xData):
    embeddings = CountVectorizer(tokenizer=lambda doc: doc, min_df=2).fit_transform(xData)
    embeddings = embeddings.toarray()
    return embeddings

x_train, y_train, x_test, y_test = rf.createTrainTestData()

x_train = createEmbeddings(x_train)
x_test = createEmbeddings(x_test)
x_train = np.reshape(x_train, (x_train.shape[0], len(x_train[0])))/255.
x_test =  np.reshape(x_test, (x_test.shape[0], len(x_test[0])))/255.
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


# print(len(x_test), "llll")
# print("\n")
# for i in x_test:
#     print(len(i))
# for i in range(len(x_train)):
#     # print(len(i))
#     x = x_train[i]
#     np.pad(x, (1,1), 'constant', constant_values=(0,0))
#     print(len(x))
# print(len(x_train))


# rnn  = RNN.RNN([len(x_train), 128, 128, len(y_train)])


# batch_size = 50
# epochs = 10
# steps_per_epoch = int(x_train.shape[0]/5)
# lr = 0.5

# print('Steps per epoch:', steps_per_epoch)

# history = rnn.train(
#     x_train, y_train,
#     x_test, y_test,
#     epochs, steps_per_epoch,
#     batch_size, lr
# )


# rnn = RNN.RNN(10, 0.3, 10, [len(x_train[0]), 10, 10, 5]) 

# stepsPerEpoch = int(x_train.shape[0]/5)
# print(x_train.shape[0])
# print("stepsPerEpoch: ", stepsPerEpoch)
# measures = rnn.trainModel(x_train, y_train, x_test, y_test, stepsPerEpoch)

# rnn = RNN.RNN([len(x_train[0]), 5, 5])
# rnn.info()

# batch_size = 10
# epochs = 10
# steps_per_epoch = 4 #nt(x_train.shape[0]/5)
# lr = 0.3

# print('Steps per epoch:', steps_per_epoch)

# history = rnn.train(
#     x_train, y_train,
#     x_test, y_test,
#     epochs, steps_per_epoch,
#     batch_size, lr
# )