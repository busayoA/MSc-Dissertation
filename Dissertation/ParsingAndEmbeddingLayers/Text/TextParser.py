import os
import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from os.path import dirname, join

merge = "./Datasets/Merge Sort"
quick = "./Datasets/Quick Sort"

currentDirectory = dirname(__file__) #the current working directory on the device
pathSplit = "/ParsingAndEmbeddingLayers"
head = currentDirectory.split(pathSplit) #split the path into two separate parts
path = head[0] 
# print(path)

merge = join(path, merge) #join the directory path to the absolute path
quick = join(path, quick)

class TextParser:
    def readTextFile(self, filePath):
        """
        Read the contents of a file
        filePath - THe file to be read

        Returns
        f.read() - The contents of the file in 'filePath'
        """
        with open(filePath, 'r') as f:
            return f.read()

    def assignTextLabels(self, filePath, fileList, labelList):
        """
        Assign class labels to each file based on the sorting algorithm it implements
        filePath - The path of the file to be read
        fileList - The list to save the read files into
        labelList - The list to save the class labels into
        """
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
        """
        Call the assignTextLabels method and splut the data into training and testing data

        Returns
        x_train - The training data
        y_train - The training data labels
        x_test - The testing data
        y_test - The testing data labels
        """
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
        """
        Run vectorization on the training and testing data

        Returns
        x_train - The vectorized form of the training data
        y_train - The training data labels
        x_test - The vectorized form of the testing data
        y_test - The testing data labels        
        """
        x_train, y_train, x_test, y_test = self.getTextData()

        # use the Scikit Learn vectorizer to fit the data on the training set
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
