import nltk, string, csv, random
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import twitter_samples
nltk.download('twitter_samples')

positiveX = twitter_samples.strings('positive_tweets.json')
negativeX = twitter_samples.strings('negative_tweets.json')

# allTweets = positiveX + negativeX
# random.shuffle(allTweets)
# split = int(0.7*len(allTweets))
# print(split)

# xTrain = allTweets[:split]
# xTest = allTweets[split:]

def saveData(xData, positives, xFile, yFile):
    yData = []
    with open(xFile, 'w', encoding='UTF8') as file:
        writer = csv.writer(file)
        writer.writerow(['tweet'])
        for i in range(len(xData)):
            xData[i] = xData[i].lower()
            text = "".join([char for char in xData[i] if char not in string.punctuation])
            text = nltk.word_tokenize(text)
            if xData[i] in positives:
                yData.append(0)
            else:
                yData.append(1)
            text = " ".join([char for char in text if not char.startswith('http')])
            writer.writerow([text])

    with open(yFile, 'w', encoding='UTF8') as file:
        writer = csv.writer(file)
        writer.writerow(['label'])
        for i in range(len(yData)):
            writer.writerow([yData[i]])

    return xData, yData

# x_train, y_train = saveData(xTrain, positiveX, 'trainTweets.csv', 'trainLabels.csv')
# x_test, y_test = saveData(xTest, positiveX, 'testTweets.csv', 'testLabels.csv')

xTrain = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/trainTweets.csv", encoding = "UTF8")
xTest = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/testTweets.csv", encoding = "UTF8")

yTrain = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/trainLabels.csv", encoding = "UTF8")
yTest = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/testLabels.csv", encoding = "UTF8")

xTrain = list(xTrain['tweet'])
xTest = list(xTest['tweet'])
yTrain = list(yTrain['label'])
yTest = list(yTest['label'])

# print(len(xTrain), len(yTrain), len(xTest), len(yTest))

# Vectorize the training and testing data

def getData():
    return xTrain, yTrain, xTest, yTest


def getVectorizedData():
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




