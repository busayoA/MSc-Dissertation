import nltk, string, csv, random, re
import pandas as pd
from sklearn import neural_network
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from cleantext import clean
from nltk.corpus import twitter_samples
# nltk.download('twitter_samples')

alphabetLower = list(string.ascii_lowercase)
alphabetUpper = list(string.ascii_uppercase)
digits = list(string.digits)


positives = twitter_samples.strings('positive_tweets.json')
negatives = twitter_samples.strings('negative_tweets.json')
neutrals = list(pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/neutralTweets.csv"))
nLabels = [1] * len(neutrals)


allTweets = positives + negatives + neutrals
random.shuffle(allTweets)
split = int(0.7*len(allTweets))
print(split)

xTrain = allTweets[:split]
xTest = allTweets[split:]


# -======================================================================================================================
# t = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/tweets.csv", encoding = "latin-1")
# # print(t[0])
# t = t.set_axis(['Sentiment', 'ID', 'Date', 'No_Query', 'User', 'Tweet'], axis=1, inplace=False)
# t = shuffle(t)
# # print(t.head())


# allTweets = list(t['Tweet'])
# allSentiments = list(t['Sentiment'])
# nsplit = int(0.7*len(neutrals))
# nLabels = [1] * len(neutrals)

# nTrainX = neutrals[:nsplit]
# nTestX = neutrals[nsplit:]

# nTrainY = nLabels[:nsplit]
# nTestY = nLabels[nsplit:]


# split = int(0.7*len(allTweets))
# print(split)

# xTrain = allTweets[:split]
# xTest = allTweets[split:]

# xTrain = xTrain + nTrainX
# xTest = xTest + nTestX

# yTrain = allSentiments[:split]
# yTest = allSentiments[split:]

# yTrain = yTrain + nTrainY
# yTest = yTest + nTestY

# print(yTest)

def saveData(xData, xFile, yFile):
    yData = []
    with open(xFile, 'w', encoding='UTF8') as file:
        writer = csv.writer(file)
        writer.writerow(['tweet'])
        for i in range(len(xData)):
            if xData[i] in positives:
                yData.append(0)
            elif xData[i] in neutrals:
                yData.append(1)
            elif xData[i] in negatives:
                yData.append(2)
            xData[i] = xData[i].lower()
            text = "".join([char for char in xData[i] if char not in string.punctuation])
            text = clean(text, no_emoji=True)
            text = nltk.word_tokenize(text)
            text = " ".join([char for char in text if not char.startswith('http')])
            writer.writerow([text])

    with open(yFile, 'w', encoding='UTF8') as file:
        writer = csv.writer(file)
        writer.writerow(['label'])
        for i in range(len(yData)):
            writer.writerow([yData[i]])

    return xData, yData

# x_train, y_train = saveData(xTrain, 'trainTweets.csv', 'trainLabels.csv')
# x_test, y_test = saveData(xTest, 'testTweets.csv', 'testLabels.csv')
# print(len(x_train), len(y_train), len(x_test), len(y_test))


xTrain = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/trainTweets.csv", encoding = "UTF8")
xTest = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/testTweets.csv", encoding = "UTF8")

yTrain = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/trainLabels.csv", encoding = "UTF8")
yTest = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/testLabels.csv", encoding = "UTF8")

xTrain = list(xTrain['tweet'])
xTest = list(xTest['tweet'])
yTrain = list(yTrain['label'])
yTest = list(yTest['label'])

xTrain = [str(line) for line in xTrain]
xTest = [str(line) for line in xTest]
# for i in range(len(yTrain)):
#     if yTrain[i] == 4:
#         yTrain[i] = 2

# for i in range(len(yTest)):
#     if yTest[i] == 4:
#        yTest[i] = 2

# print(yTrain)

# print(len(xTrain), len(yTrain), len(xTest), len(yTest))

# Vectorize the training and testing data

def getData():
    return xTrain, yTrain, xTest, yTest




def getVectorizedData():
    xTrain, yTrain, xTest, yTest = getData()

    vectorizer = CountVectorizer(tokenizer=lambda doc:doc, min_df=2)
    vectorizer.fit(xTrain)
    x_train = vectorizer.transform(xTrain)
    x_train = x_train.toarray()/255.
    # x_train = np.reshape(x_train, (x_train.shape[0], len(x_train[0])))  
    x_test  = vectorizer.transform(xTest)
    x_test = x_test.toarray()/255.
    # x_test =  np.reshape(x_test, (x_test.shape[0], len(x_test[0]))) 
    y_train = tf.keras.utils.to_categorical(yTrain)
    y_test = tf.keras.utils.to_categorical(yTest)

    return x_train, y_train, x_test, y_test

getVectorizedData()
# # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # GET THE DATA/TEXT 
# # with open("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Corona_NLP_train.csv") as file:
# trainingData = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Corona_NLP_train.csv", encoding = "ISO-8859-1")
# testingData = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Corona_NLP_test.csv", encoding = "ISO-8859-1")

# sentimentLabel = {'Extremely Negative': 0, 'Negative': 1, 'Neutral': 2, 'Positive': 3, 'Extremely Positive': 4}
# trainingData['Sentiment'] =  [sentimentLabel[item] for item in trainingData['Sentiment']]
# testingData['Sentiment'] =  [sentimentLabel[item] for item in testingData['Sentiment']]

# trainX = trainingData['OriginalTweet']
# trainY = trainingData['Sentiment']

# testX = testingData['OriginalTweet']
# testY = testingData['Sentiment']

# finalY = []
# def cleanData(xData, yData, fileToWriteX, fileToWriteY):
#     y = []
#     with open(fileToWriteX, 'w', encoding='UTF8') as file:
#         writer = csv.writer(file)
#         writer.writerow(['tweet'])
#         for i in range(len(xData)):
#             xData[i] = xData[i].lower()
#             text = "".join([char for char in xData[i] if char not in string.punctuation])
#             text = nltk.word_tokenize(text)
#             if len(text) > 10:
#                 y.append(yData[i])
#                 text = " ".join([char for char in text if not char.startswith('http')])
#                 print(text)
#                 writer.writerow([text])
    
#     with open(fileToWriteY, 'w', encoding='UTF8') as file:
#         writer = csv.writer(file)
#         writer.writerow(['tweet'])
#         for i in range(len(y)):
#             writer.writerow([y[i]])

#     return xData, y

# # train, yTrain = cleanData(trainX, trainY, 'cleanTrain.csv', 'yTrain.csv')
# # test, yTest = cleanData(testX, testY, 'cleanTest.csv', 'yTest.csv')

# # xTrain = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/cleanTrain.csv", encoding = "UTF8")
# # xTest = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/cleanTest.csv", encoding = "UTF8")

# # yTrain = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/yTrain.csv", encoding = "UTF8")
# # yTest = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/yTest.csv", encoding = "UTF8")

# # xTrain = list(xTrain['tweet'])
# # xTest = list(xTest['tweet'])
# # yTrain = list(yTrain['tweet'])
# # yTest = list(yTest['tweet'])

# # # print(len(xTrain), len(yTrain), len(xTest), len(yTest))

# # def getData():
# #     return xTrain, yTrain, xTest, yTest


# for i in range(25, 101):
#     fileName = '{}'.format(i)
#     fileName = fileName + '.py'
#     print(fileName)
#     with open(fileName, 'w') as f:
#         f.write("")
