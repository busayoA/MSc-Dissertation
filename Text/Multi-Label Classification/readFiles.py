import os
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer

merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort"
quick = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Quick Sort"
other = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Other"

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
    mergeList, mergeLabels, quickList, quickLabels, otherList, otherLabels  = [], [], [], [], [], [] #the training and testing data
    # create training data:
    assignLabels(merge, mergeList, mergeLabels)
    assignLabels(quick, quickList, quickLabels)
    assignLabels(other, otherList, otherLabels)

    x_train, y_train, x_test, y_test = [], [], [], []

    x_train = mergeList[:int(0.7*len(mergeList))] + quickList[:int(0.7*len(quickList))] + otherList[:int(0.7*len(otherList))]
    x_test = mergeList[int(0.7*len(mergeList)):] + quickList[int(0.7*len(quickList)):] + otherList[int(0.7*len(otherList)):]

    y_train = mergeLabels[:int(0.7*len(mergeList))] + quickLabels[:int(0.7*len(quickList))] + otherLabels[:int(0.7*len(otherLabels))]
    y_test = mergeLabels[int(0.7*len(mergeList)):] + quickLabels[int(0.7*len(quickList)):] + otherLabels[int(0.7*len(otherLabels)):]

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
    # x_train = np.reshape(x_train, (x_train.shape[0], len(x_train[0])))  

    x_test  = vectorizer.transform(x_test)
    x_test = x_test.toarray()/255.
    y_test = tf.keras.utils.to_categorical(y_test)
    # x_test =  np.reshape(x_test, (x_test.shape[0], len(x_test[0]))) 

    return x_train, y_train, x_test, y_test

