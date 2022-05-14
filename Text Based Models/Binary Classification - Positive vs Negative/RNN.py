import nltk, string
import numpy as np
import pandas as pd
import readFiles as rf
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords


stopWords = stopwords.words('english')

amazon = pd.read_csv("Text Data/amazon_cells_labelled.txt", names=['review', 'sentiment'], sep='\t')
yelp = pd.read_csv("Text Data/yelp_labelled.txt", names=['review', 'sentiment'], sep='\t')

dataSets = [amazon, yelp]
allReviews = pd.concat(dataSets)
xData = list(allReviews['review'])
yData = list(allReviews['sentiment'])

vocabulary = []

for sentence in xData:
    sentence = nltk.word_tokenize(sentence)
    for word in sentence:
        word = word.lower()
        if word not in vocabulary and word not in stopWords and word not in string.punctuation:
            vocabulary.append(word)

print(vocabulary)
    
# print(x)

# xTrain = xData[:int(0.5*len(xData))]
# yTrain = yData[:int(0.5*len(xData))]

# xTest = xData[int(0.5*len(xData)):]
# yTest = yData[int(0.5*len(xData)):]



# # Generate a range of sequences (based on the sequenceLength variable) to use for 
# # classifying the text. The sequence length defines the number of words that will be 
# # considered in context with each other when analysing the data and building the model

# vocab = []

# for word in xData:
#     if word not in vocab:
#         vocab.append(word)
# # vDict = dict((i, v) for v, i in enumerate(vocabulary))
# # x_train, y_train, x_test, y_test = [], [], [], []
# print(vocab)

# def vectorizer(xValues, yValues, embedLength):
#     x, y = [], []
#     for i in range(0, len(xValues)-embedLength):
#         input = xTrain[i:i + embedLength]
#         x.append([vDict[j] for j in input])
#         y.append(yValues[i])

#     return x, y

# embedLength = 100
# x_train, y_train = vectorizer(xTrain, yTrain, embedLength)
# x_test, y_test = vectorizer(xTest, yTest, embedLength)

# x_train, x_test = np.array(x_train)/float(len(vocabulary)), np.array(x_test)/float(len(vocabulary)) 

# x_train = np.reshape(x_train, (x_train.shape[0], len(x_train[0]))) 
# y_train = tf.keras.utils.to_categorical(y_train)    


# x_test = np.reshape(x_test, (x_test.shape[0], len(x_test[0]))) 
# x_test = x_test/float(len(vocabulary)) 
# y_test = tf.keras.utils.to_categorical(y_test)    

# # print(y_test)
# # # x_train, y_train, x_test, y_test = rf.getVectorizedTextData()
# filename = "textRNNModel.hdf5"

# model = Sequential()
# model.add(Input(shape=(x_train.shape[0], embedLength)))

# model.add(Bidirectional(LSTM(128, input_shape=(x_train.shape[0], embedLength), activation='relu', return_sequences=True)))
# model.add(Dropout(0.2))

# model.add(LSTM(128, activation='relu'))
# model.add(Dropout(0.2))


# model.add(Dense(2, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())

# model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))
# model.save(filename)

