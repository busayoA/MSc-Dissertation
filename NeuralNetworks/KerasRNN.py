
import numpy as np
import nltk, string, csv
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils as nUtils
from keras.models import Sequential
from keras.layers import Input, Activation, Dense, Dropout, LSTM, Embedding, add, Bidirectional

stemmer = PorterStemmer()

# GET THE DATA/TEXT 
# with open("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Corona_NLP_train.csv") as file:
trainingData = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Corona_NLP_train.csv", encoding = "ISO-8859-1")
testingData = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Corona_NLP_test.csv", encoding = "ISO-8859-1")


sentimentLabel = {'Extremely Negative': 0, 'Negative': 1, 'Neutral': 2, 'Positive': 3, 'Extremely Positive': 4}
trainingData['Sentiment'] =  [sentimentLabel[item] for item in trainingData['Sentiment']]
testingData['Sentiment'] =  [sentimentLabel[item] for item in testingData['Sentiment']]

xTrain = trainingData['OriginalTweet']
yTrain = trainingData['Sentiment']

xTest = testingData['OriginalTweet']
yTest = testingData['Sentiment']

def cleanData(xData, fileToWrite):
    with open(fileToWrite, 'w', encoding='UTF8') as file:

        writer = csv.writer(file)
        for i in range(len(xData)):
            xData[i] = xData[i].lower()
            text = "".join([char for char in xData[i] if char not in string.punctuation])
            text = nltk.word_tokenize(text)
            text = [word for word in text if word not in nltk.corpus.stopwords.words('english')]
            text = [stemmer.stem(word) for word in text]
            text = " ".join([char for char in text])
            xData[i] = text
            print(text)

            writer.writerow([text])
    return xData

# xTrain = cleanData(xTrain, 'cleanTrain.csv')
# xTest = cleanData(xTest, 'cleanTest.csv')

xTrain = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/cleanTrain.csv", encoding = "UTF8")
xTest = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/cleanTest.csv", encoding = "UTF8")

xTrain = list(xTrain['originalTweet'])
xTest = list(xTest['originalTweet'])

vectorizer = CountVectorizer(tokenizer=lambda doc: doc, min_df=2)
vectorizer.fit(xTrain)
seq = vectorizer.transform(xTrain)
sequenceLength = seq.shape[1]
seqY = tf.keras.utils.to_categorical(yTrain)
seqTest = vectorizer.transform(xTest)
seqTestY = tf.keras.utils.to_categorical(yTest)

seq.toarray()
seqTest.toarray()
seq = seq/255.
seqTest = seqTest/255.
seq = np.reshape(seq, (seq.shape[0], sequenceLength))
seqTest =  np.reshape(seqTest, (seqTest.shape[0], sequenceLength))

def vectoriser(xData, yData, seqLength):
    vocab = sorted(set(xData))
    vectorized = dict((i, v) for v, i in enumerate(vocab))
    # print(vectorizedTest)
    x, y = np.empty([len(xData)-seqLength, seqLength]), []
    for i in range(0, len(xData)-seqLength):
        inputSequence = xData[i:i+seqLength]
        x[i] = np.asarray([vectorized[char] for char in inputSequence])
        y.append(yData[i])

    # print(x_train)
    # print(y_train) 
    x, y = np.reshape(x, (len(x), seqLength)),  nUtils.to_categorical(y) 

    x = x/float(len(vocab)) #convert all values to floats
    return x, y, len(vocab)

x_train, y_train, inputDim = vectoriser(xTrain, yTrain, sequenceLength)
x_test, y_test, testInputDim = vectoriser(xTest, yTest, sequenceLength)
# print(x_train)
# print()
# print(y_train)


# Sequential model -> the layers are stacked linearly
model = Sequential()

filename = "firstTry.hdf5"

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# The input layer 
# model.add(Input(shape=(sequenceLength, 1)))
# model.add(LSTM(256, input_shape=(sequenceLength, 1), dropout=0.3, return_sequences=True))
# model.add(LSTM(128))
# model.add(Dense(5, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# print(model.summary())
# model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
# model.save(filename)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------



# model.add(Embedding(input_dim = inputDim, output_dim = 256))
# model.add(Input(shape=(71, )))
# model.add(LSTM(256, input_shape=(71, ), dropout=0.3, return_sequences=True))
# model.add(LSTM(128))
# model.add(Dense(5, activation='softmax'))
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# model.fit(seq, seqY, epochs=5, batch_size=256, validation_data=(seqTest, seqTestY))
# model.save(filename)

# vocab = sorted(set(xTrain))
# vectorizedTrain = dict((i, v) for v, i in enumerate(vocab))
# sequenceLength = 1000
# x_train, y_train, x_test, y_test  = np.empty([len(xTrain)-sequenceLength, sequenceLength]), [], np.empty([len(xTest)-sequenceLength, sequenceLength]), [] 

# for i in range(0, len(xTrain)):
#     inputSequence = xTrain[i:i+sequenceLength]
#     outputSequence = xTrain[i + sequenceLength]

#     x_train[i] = np.asarray([vectorizedTrain[char] for char in inputSequence])
#     y_train.append(vectorizedTrain[outputSequence])

# # print(x_train)
# # x_train = np.reshape(x_train, (len(x_train), 1)) 
# # x_train = np.array(x_train)
# x_train = x_train/float(len(vocab)) #convert all values to floats
# # print(x_train)

# testVocab = sorted(set(xTest))
# vectorizedTest = dict((i, v) for v, i in enumerate(testVocab))
# for i in range(0, len(xTest)-sequenceLength):
#     inputSequence = xTest[i:i+sequenceLength]
#     outputSequence = xTest[i + sequenceLength]

#     x_test[i] = np.asarray([vectorizedTest[char] for char in inputSequence])
#     y_test.append(vectorizedTest[outputSequence])
# # print(x_test)
# # x_test = np.array(x_test)
# # x_test = np.reshape(x_test, (len(x_test), sequenceLength)) 


# x_test = x_test/float(len(testVocab)) #convert all values to floats
# # print(x_test) 
# y_train = nUtils.to_categorical(y_train) 
# y_test = nUtils.to_categorical(y_test) 
# print(y_train)


# # vectorizer = CountVectorizer(tokenizer=lambda doc: doc, min_df=2)
# # vectorizer.fit(xTrain)

# # x_train = vectorizer.transform(xTrain)
# # x_test  = vectorizer.transform(xTest)



# # x_train, y_train = x_train.toarray(), tf.keras.utils.to_categorical(yTrain, num_classes=5) 
# # x_test, y_test = x_test.toarray(), tf.keras.utils.to_categorical(yTest, num_classes=5) 

# # x_train = np.reshape(x_train, (x_train.shape[0], len(x_train[0]), 1))
# # x_train = x_train/float(len(x_train)) #convert all values to floats

# # x_test = np.reshape(x_test, (x_test.shape[0], len(x_test[0]), 1))
# # x_test = x_test/float(len(x_test)) #convert all values to floats


# # Sequential model -> the layers are stacked linearly
# model = Sequential()

# filename = "firstTry.hdf5"

# # The input layer
# model.add(LSTM(256, input_shape=(1000, 1), dropout=0.3, return_sequences=True))
# model.add(LSTM(128, return_sequences=True))
# model.add(Dense(5, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# checkpoint = ModelCheckpoint(filename, monitor='loss', verbose=1, save_best_only=True, mode='min')
# print(model.summary())
# model.fit(x_train, y_train, epochs=20, batch_size=256, validation_data=(x_test, y_test))
# model.save(filename)



# # batchSize = 64
# # model = tf.keras.Sequential([
# #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
# #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
# #     tf.keras.layers.Dense(64, activation='relu'),
# #     tf.keras.layers.Dropout(0.5),
# #     tf.keras.layers.Dense(1)
# # ])

# # print([layer.supports_masking for layer in model.layers])

# # predictions = model.predict(np.array([x_train[:1]]))
# # print(predictions[0])


def nbClassify(trainSet, trainCategories, testSet):
    """A MULTINOMINAL NB CLASSIFIER"""
    # Build a pipeline to simplify the process of creating the vector matrix, transforming to tf-idf and classifying
    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
    # text_clf.fit(twenty_train.data, twenty_train.target)
    text_clf.fit(trainSet, trainCategories)

    # Evaluating the performance of the classifier

    predictions = text_clf.predict(testSet)
    return predictions


# CLASSIFYING USING AN SVM (SUPPORT VECTOR MACHINE)
def svmClassify(trainSet, trainCategories, testSet):
    """AN ALTERNATIVE TO THE MULTINOMINAL NB CLASSIFIER"""
    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None))])
    text_clf.fit(trainSet, trainCategories)

    predictions = text_clf.predict(testSet)
    return predictions

pred = nbClassify(xTrain, yTrain, xTest)

# pred = svmClassify(xTrain, yTrain, xTest)
accuracy = 0

for i in range(len(pred)):
    if pred[i] == yTest[i]:
        accuracy += 1
    
print(accuracy/len(yTest))
