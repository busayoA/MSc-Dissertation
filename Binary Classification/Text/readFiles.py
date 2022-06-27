import os
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from os.path import dirname, join

current_dir = dirname(__file__)
merge = "./Data/Merge Sort"
quick = "./Data/Quick Sort"

merge = join(current_dir, merge)
quick = join(current_dir, quick)

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
    y_train = tf.keras.utils.to_categorical(y_train)

    x_test  = vectorizer.transform(x_test)
    x_test = x_test.toarray()/255.
    y_test = tf.keras.utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test

getVectorizedCodeData()
