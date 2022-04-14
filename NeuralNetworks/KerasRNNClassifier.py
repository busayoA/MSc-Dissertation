import nltk
import numpy as np
import tensorflow as tf
import readTextFiles as rtf
import matplotlib.pyplot as plt
from keras.utils import np_utils as nUtils
from keras.models import Sequential
from keras.layers import Input, Activation, Dense, Dropout, LSTM, Embedding, add, Bidirectional
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import twitter_samples
nltk.download('twitter_samples')

# vectorizer = CountVectorizer(tokenizer=lambda doc: doc, min_df=2)
# vectorizer.fit(xTrain)
# seq = vectorizer.transform(xTrain)
# sequenceLength = seq.shape[1]
# seqY = tf.keras.utils.to_categorical(yTrain)
# seqTest = vectorizer.transform(xTest)
# seqTestY = tf.keras.utils.to_categorical(yTest)
# seq.toarray()
# seqTest.toarray()
# seq = seq/255.
# seqTest = seqTest/255.
# seq = np.reshape(seq, (seq.shape[0], sequenceLength))
# seqTest =  np.reshape(seqTest, (seqTest.shape[0], sequenceLength))

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# print(all_positive_tweets)

def vectoriser(xData, yData, seqLength):
    vocab = sorted(set(xData))
    vectorized = dict((i, v) for v, i in enumerate(vocab))
    # print(vectorizedTest)
    x, y = np.empty([len(xData)-seqLength, seqLength]), []
    for i in range(0, len(xData)-seqLength):
        inputSequence = xData[i:i+seqLength]
        x[i] = np.asarray([vectorized[char] for char in inputSequence])
        y.append(yData[i])
    x, y = np.reshape(x, (len(x), seqLength)),  nUtils.to_categorical(y) 
    x = x/float(len(vocab)) #convert all values to floats
    return x, y

xTrain, yTrain, xTest, yTest = rtf.getData()

sequenceLength = 100
x_train, y_train = vectoriser(xTrain, yTrain, sequenceLength)
x_test, y_test = vectoriser(xTest, yTest, sequenceLength)

model = Sequential()
filename = "firstTry.hdf5"
# # ----------------------------------------------------------------------------------------------------------------------------------------------------------
model.add(Input(shape=(sequenceLength, 1)))
model.add(Bidirectional(LSTM(256, input_shape=(sequenceLength, 1), return_sequences=True, activation='relu')))
model.add(LSTM(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, epochs=20, batch_size=256, validation_data=(x_test, y_test))
model.save(filename)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

