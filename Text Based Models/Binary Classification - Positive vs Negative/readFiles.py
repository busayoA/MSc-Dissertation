import random
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import twitter_samples

# # =====================================================================================================================================================================================
# # CODE 
# # select the set of positive and negative tweets
positiveX = twitter_samples.strings('positive_tweets.json')
negativeX = twitter_samples.strings('negative_tweets.json')

# concatenate the lists, 1st part is the positive tweets followed by the negative
x = positiveX + negativeX
random.shuffle(x)

y = []
for i in x:
    if i in negativeX:
        y.append(0)
    else:
        y.append(1)

# split = int(0.7*len(x))


# x_train = x[:split]
# y_train = y[:split]

# x_test = x[split:]
# y_test = y[split:]

# def saveData(xData, xFile, yFile):
#     yData = []
#     with open(xFile, 'w', encoding='UTF8') as file:
#         writer = csv.writer(file)
#         writer.writerow(['tweet'])
#         for i in range(len(xData)):
#             text = xData[i].lower()
#             text = "".join([char for char in xData[i] if char not in string.punctuation])
#             text = nltk.word_tokenize(text)
#             if len(text) > 2:
#                 text = " ".join([char for char in text if not char.startswith('http')])
#                 if text.strip():
#                     writer.writerow([text])
#                     if xData[i] in positiveX:
#                         yData.append(0)
#                     elif xData[i] in negativeX:
#                         yData.append(1)

#     with open(yFile, 'w', encoding='UTF8') as file:
#         writer = csv.writer(file)
#         writer.writerow(['label'])
#         for i in range(len(yData)):
#             writer.writerow([yData[i]])

# saveData(x_train, 'x_train.csv', 'y_train.csv')
# saveData(x_test, 'x_test.csv', 'y_test.csv')

def getVectorizedTextData():
    amazon = pd.read_csv("Text Data/amazon_cells_labelled.txt", names=['review', 'sentiment'], sep='\t')
    imdb = pd.read_csv("Text Data/imdb_labelled.txt", names=['review', 'sentiment'], sep='\t')
    yelp = pd.read_csv("Text Data/yelp_labelled.txt", names=['review', 'sentiment'], sep='\t')

    dataSets = [amazon, imdb, yelp]
    allReviews = pd.concat(dataSets)
    x_train = list(allReviews['review'])
    y_train = list(allReviews['sentiment'])

    x_test = x
    y_test = y

    vocabulary = sorted(set(x_train))

    vectorizer = CountVectorizer(tokenizer=lambda doc:doc, min_df=2, vocabulary=vocabulary)
    vectorizer = vectorizer.fit(x_train)

    x_train = vectorizer.transform(x_train)
    x_train = x_train.toarray()/255.
    x_train = np.reshape(x_train, (x_train.shape[0], len(x_train[0]))) 
    y_train = tf.keras.utils.to_categorical(y_train)    

    x_test = vectorizer.transform(x_test)
    x_test = x_test.toarray()/255.
    x_test =  np.reshape(x_test, (x_test.shape[0], len(x_test[0]))) 
    y_test = tf.keras.utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test

