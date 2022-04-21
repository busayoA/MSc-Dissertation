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
pythonFiles = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Languages/Python"
javaFiles = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Languages/Java"
cFiles = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Languages/C"

def readFile(filePath):
    with open(filePath, 'r') as f:
        return f.read()

def assignLabels(filePath, fileList, labelList):
    os.chdir(filePath)
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".py") or file.endswith(".java") or file.endswith(".c"):
            path = f"{filePath}/{file}"
            # call read text file function
            fileList.append(readFile(path))
            if filePath.find("Python") != -1:
                labelList.append(0)
            elif filePath.find("Java") != -1:
                labelList.append(1)
            else:
                labelList.append(2)

def getCodeData():
    pList, pLabels, jList, jLabels, cList, cLabels  = [], [], [], [], [], [] #the training and testing data
    # create training data:
    assignLabels(pythonFiles, pList, pLabels)
    assignLabels(javaFiles, jList, jLabels)
    assignLabels(cFiles, cList, cLabels)

    x_train, y_train, x_test, y_test = [], [], [], []

    x_train = pList[:int(0.7*len(pList))] + jList[:int(0.7*len(jList))] + cList[:int(0.7*len(cList))]
    x_test = pList[int(0.7*len(pList)):] + jList[int(0.7*len(jList)):] + cList[int(0.7*len(cList)):]

    y_train = pLabels[:int(0.7*len(pLabels))] + jLabels[:int(0.7*len(jLabels))] + cLabels[:int(0.7*len(cLabels))]
    y_test = pLabels[int(0.7*len(pLabels)):] + jLabels[int(0.7*len(jLabels)):] + cLabels[int(0.7*len(cLabels)):]

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

