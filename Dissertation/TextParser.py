import os
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from os.path import dirname, join

current_dir = dirname(__file__)
merge = "./Data/Merge Sort"
quick = "./Data/Quick Sort"

merge = join(current_dir, merge)
quick = join(current_dir, quick)


"""PARSING THE FILES AS TEXT"""
class TextParser:
    def readTextFile(self, filePath):
        with open(filePath, 'r') as f:
            return f.read()

    def assignTextLabels(self, filePath, fileList, labelList):
        os.chdir(filePath)
        for file in os.listdir():
            # Check whether file is in text format or not
            if file.endswith(".py"):
                path = f"{filePath}/{file}"
                # call read text file function
                fileList.append(self.readTextFile(path))
                if filePath.find("Merge") != -1:
                    labelList.append(0)
                elif filePath.find("Quick") != -1:
                    labelList.append(1)

    def getTextData(self):
        mergeList, mergeLabels, quickList, quickLabels  = [], [], [], [] #the training and testing data
        self.assignTextLabels(merge, mergeList, mergeLabels)
        self.assignTextLabels(quick, quickList, quickLabels)
        x_train, y_train, x_test, y_test = [], [], [], []
        x_train = mergeList[:int(0.7*len(mergeList))] + quickList[:int(0.7*len(quickList))] 
        x_test = mergeList[int(0.7*len(mergeList)):] + quickList[int(0.7*len(quickList)):]
        y_train = mergeLabels[:int(0.7*len(mergeList))] + quickLabels[:int(0.7*len(quickList))]
        y_test = mergeLabels[int(0.7*len(mergeList)):] + quickLabels[int(0.7*len(quickList)):] 

        return x_train, y_train, x_test, y_test

    def getVectorizedTextData(self):
        x_train, y_train, x_test, y_test = self.getTextData()

        vectorizer = CountVectorizer(tokenizer=lambda doc:doc, min_df=2)
        vectorizer.fit(x_train)
        x_train = vectorizer.transform(x_train)
        x_train = x_train.toarray()/255.
        x_train = tf.convert_to_tensor(x_train,  dtype=np.float32)
        y_train = tf.keras.utils.to_categorical(y_train)

        x_test  = vectorizer.transform(x_test)
        x_test = x_test.toarray()/255.
        x_test = tf.convert_to_tensor(x_test, dtype=np.float32)
        y_test = tf.keras.utils.to_categorical(y_test)

        return x_train, y_train, x_test, y_test
