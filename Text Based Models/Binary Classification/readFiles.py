import nltk, string, csv, random, os
import pandas as pd
from sklearn import neural_network
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from cleantext import clean
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords

# =====================================================================================================================================================================================
# CODE 
merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort"
quick = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Quick Sort"

def readFile(filePath):
    with open(filePath, 'r') as f:
        return f.read()

def assignLabels(filePath, fileList, labelList):
    os.chdir(filePath)
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".py"):
            path = f"{filePath}/{file}"
            # call read text file function
            fileList.append(readFile(path))
            if filePath.find("Merge") != -1:
                labelList.append(0)
            elif filePath.find("Quick") != -1:
                labelList.append(1)
            elif filePath.find("Other") != -1:
                labelList.append(2)

def getCodeData():
    mergeList, mergeLabels, quickList, quickLabels  = [], [], [], [] #the training and testing data
    # create training data:
    assignLabels(merge, mergeList, mergeLabels)
    assignLabels(quick, quickList, quickLabels)

    x_train, y_train, x_test, y_test = [], [], [], []

    x_train = mergeList[:int(0.7*len(mergeList))] + quickList[:int(0.7*len(quickList))] 
    x_test = mergeList[int(0.7*len(mergeList)):] + quickList[int(0.7*len(quickList)):]
    y_train = mergeLabels[:int(0.7*len(mergeList))] + quickLabels[:int(0.7*len(quickList))]
    y_test = mergeLabels[int(0.7*len(mergeList)):] + quickLabels[int(0.7*len(quickList)):] 

    return x_train, y_train, x_test, y_test

# x_train, y_train, x_test, y_test = getData()
# print(y_test)

def getVectorizedCodeData():
    x_train, y_train, x_test, y_test = getCodeData()
    vectorizer = CountVectorizer(tokenizer=lambda doc:doc, min_df=2)
    vectorizer.fit(x_train)
    x_train = vectorizer.transform(x_train)
    x_train = x_train.toarray()/255.
    # x_train = np.reshape(x_train, (x_train.shape[0], len(x_train[0])))  
    x_test  = vectorizer.transform(x_test)
    x_test = x_test.toarray()/255.
    # x_test =  np.reshape(x_test, (x_test.shape[0], len(x_test[0]))) 
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test

# getVectorizedCodeData()


# stop_words = stopwords.words('english')
# words = set(nltk.corpus.words.words())

# # =====================================================================================================================================================================================
# # TEXT 
# positives = list(twitter_samples.strings('positive_tweets.json'))
# negatives = list(twitter_samples.strings('negative_tweets.json'))

# allTweets = positives + negatives 
# random.shuffle(allTweets)
# split = int(0.7*len(allTweets))

# xTrain = allTweets[:split]
# xTest = allTweets[split:]

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
#                 text = [word for word in text if word in words]
#                 text = " ".join([char for char in text if not char.startswith('http')])
#                 if text.strip() :
#                     writer.writerow([text])
#                     if xData[i] in positives:
#                         yData.append(0)
#                     elif xData[i] in negatives:
#                         yData.append(1)
                

#     with open(yFile, 'w', encoding='UTF8') as file:
#         writer = csv.writer(file)
#         writer.writerow(['label'])
#         for i in range(len(yData)):
#             writer.writerow([yData[i]])

#     return xData, yData

# # x_train, y_train = saveData(xTrain, 'trainTweets.csv', 'trainLabels.csv')
# # x_test, y_test = saveData(xTest, 'testTweets.csv', 'testLabels.csv')
# # print(len(x_train), len(y_train), len(x_test), len(y_test))

# # Vectorize the training and testing data
# xTrain = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Text/Binary Classification/Data/trainTweets.csv", encoding = "UTF8")
# xTest = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Text/Binary Classification/Data/testTweets.csv", encoding = "UTF8")

# yTrain = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Text/Binary Classification/Data/trainLabels.csv", encoding = "UTF8")
# yTest = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Text/Binary Classification/Data/testLabels.csv", encoding = "UTF8")

# xTrain = list(xTrain['tweet'])
# xTest = list(xTest['tweet'])
# yTrain = list(yTrain['label'])
# yTest = list(yTest['label'])

# def getTextData():
#     return xTrain, yTrain, xTest, yTest

# def getVectorizedTextData():
#     vectorizer = CountVectorizer(tokenizer=lambda doc:doc, min_df=2)
#     vectorizer = vectorizer.fit(xTrain)
#     x_train = vectorizer.transform(xTrain)
#     x_train = x_train.toarray()/255.
#     # x_train = np.reshape(x_train, (x_train.shape[0], len(x_train[0]))) 
#     x_test = vectorizer.transform(xTest)
#     x_test = x_test.toarray()/255.
#     # x_test =  np.reshape(x_test, (x_test.shape[0], len(x_test[0]))) 
#     y_train = tf.keras.utils.to_categorical(yTrain)
#     y_test = tf.keras.utils.to_categorical(yTest)

#     return x_train, y_train, x_test, y_test


# =====================================================================================================================================================================================
