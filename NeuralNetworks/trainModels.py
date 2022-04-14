import ManualRNN as RNN
import readCodeFiles as rf
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

epochs = 10
lr = 0.2

rnn  = RNN.RNN([len(x_train), 20, 20, 5], epochs, lr)
rnn.trainModel(x_train, y_train, x_test, y_test)
for i in range(len(x_train)):
    prediction = rnn.predict(x_train[i])
    print("Expected=", y_train[i], " predicted", prediction[0][0])

for i in range(len(x_test)):
    prediction2 = rnn.predict(x_test[i])
    print("Expected=", y_test[i], " predicted", prediction2[0][0])