import random
import TreeEmbeddingLayer as embeddingLayer
import parseFiles as pf
import tensorflow as tf
from Node import Node
from os.path import dirname, join

def attachLabels(x, y):
    pairs = []
    for index in range(len(x)):
        pairs.append([x[index], y[index]])
    random.shuffle(pairs)
    return pairs

def getFileNames(padding: bool, hashed: bool):
    current_dir = dirname(__file__)
    if hashed is False:
        if padding is True:
            xTrain = join(current_dir, "./Data/x_train_padded.txt")
            yTrain = join(current_dir, "./Data/y_train_padded.txt")
            xTest = join(current_dir, "./Data/x_test_padded.txt")
            yTest = join(current_dir, "./Data/y_test_padded.txt")
        else:
            xTrain = join(current_dir, "./Data/x_train.txt")
            yTrain = join(current_dir, "./Data/y_train.txt")
            xTest = join(current_dir, "./Data/x_test.txt")
            yTest = join(current_dir, "./Data/y_test.txt")
    else:
        if padding is True:
            xTrain = join(current_dir, "./Data/x_train_hashed_padded.txt")
            yTrain = join(current_dir, "./Data/y_train_hashed_padded.txt")
            xTest = join(current_dir, "./Data/x_test_hashed_padded.txt")
            yTest = join(current_dir, "./Data/y_test_hashed_padded.txt")
        else:
            xTrain = join(current_dir, "./Data/x_train_hashed_unpadded.txt")
            yTrain = join(current_dir, "./Data/y_train_hashed_unpadded.txt")
            xTest = join(current_dir, "./Data/x_test_hashed_unpadded.txt")
            yTest = join(current_dir, "./Data/y_test_hashed_unpadded.txt")


    return xTrain, yTrain, xTest, yTest

def saveData(train, test, padding: bool):
    print("Collecting training data", end="......")
    x_train, y_train = [], []
    for i in range(len(train)):
        if i % 5 == 0:
            print(end=".")
        embedding = embeddingLayer.TreeEmbeddingLayer(train[i], padding)
        x_train.append(embedding.vectors)
        y_train.append(embedding.label)


    print("\nCollecting testing data", end="......")
    x_test, y_test = [], []
    for i in range(len(test)):
        if i % 5 == 0:
            print(end=".")
        embedding = embeddingLayer.TreeEmbeddingLayer(test[i], padding)
        x_test.append(embedding.vectors)
        y_test.append(embedding.label)
    if padding is True:
        maxLen = 311
        for i in x_train:
            if len(i) < maxLen:
                difference = maxLen - len(i)
                for j in range(difference):
                    i.append(0.0)

        for i in x_test:
            if len(i) < maxLen:
                difference = maxLen - len(i)
                for j in range(difference):
                    i.append(0.0)

    xTrain, yTrain, xTest, yTest = getFileNames(padding, False)

    with open(xTrain, 'w') as writer:
        for i in x_train:
            writer.write(str(i) + "\n")

    with open(yTrain, 'w') as writer:
        for i in y_train:
            writer.write(str(i) + "\n")    
    
    with open(xTest, 'w') as writer:
        for i in x_test:
            writer.write(str(i) + "\n")
        
    with open(yTest, 'w') as writer:
        for i in y_test:
            writer.write(str(i) + "\n")

def saveHashData(train, test, padding: bool):
    print("Collecting training data", end="......")

    if padding is True:
        maxLen = 0
        for i in train:
            if len(i[0]) > maxLen:
                maxLen = len(i[0]) 
        
        for i in test:
            if len(i[0]) > maxLen:
                maxLen = len(i[0]) #311

        for i in train:
            if len(i[0]) < maxLen:
                difference = maxLen - len(i[0])
                for j in range(difference):
                    i[0].append(0.0)

        for i in test:
            if len(i[0]) < maxLen:
                difference = maxLen - len(i[0])
                for j in range(difference):
                    i[0].append(0.0)

    x_train, y_train = [], []
    for i in range(len(train)):
        if i % 5 == 0:
            print(end=".")
        x_train.append(train[i][0])
        y_train.append(train[i][1])

    xTrain, yTrain, xTest, yTest = getFileNames(padding, True)

    with open(xTrain, 'w') as writer:
        for i in x_train:
            writer.write(str(i) + "\n")

    with open(yTrain, 'w') as writer:
        for i in y_train:
            writer.write(str(i) + "\n")
    print()
    
    print("Collecting testing data", end="......")
    x_test, y_test = [], []
    for i in range(len(test)):
        if i % 5 == 0:
            print(end=".")
        x_test.append(test[i][0])
        y_test.append(test[i][1])

    with open(xTest, 'w') as writer:
        for i in x_test:
            writer.write(str(i) + "\n")
        
    with open(yTest, 'w') as writer:
        for i in y_test:
            writer.write(str(i) + "\n")
    print()


def readXFiles(filePath):
    with open(filePath, 'r') as reader:
        values = reader.readlines()
    
    for x in range(len(values)):
        values[x] = values[x].replace("[", "").replace("]", "").strip("\n")
        values[x] = values[x].split(",")
        values[x] = [float(i) for i in values[x]]

    return values

def readYFiles(filePath):
    with open(filePath, 'r') as reader:
        values = reader.readlines()
    
    for y in range(len(values)):
        values[y] = values[y].replace("[", "").replace("]", "").strip("\n")
        values[y] = values[y].split(" ")
        values[y] = [float(i) for i in values[y]]

    return values

def getData(padding: bool, hashed: bool):
    xTrain, yTrain, xTest, yTest = getFileNames(padding, hashed)
    
    x_train, y_train, x_test, y_test = [], [], [], []
    x_train = readXFiles(xTrain)
    y_train = readYFiles(yTrain)

    x_test = readXFiles(xTest)
    y_test = readYFiles(yTest)

    return x_train, y_train, x_test, y_test

# RUN THE DATA PROCESSOR ON THE UNHASHED TREES
parser = pf.Parser()
current_dir = dirname(__file__)
merge = join(current_dir, "./Data/Merge Sort")
quick = join(current_dir, "./Data/Quick Sort")

mergeTree, mergeLabels = parser.parse(merge)
quickTree, quickLabels = parser.parse(quick)

x = mergeTree + quickTree 
y = mergeLabels + quickLabels 

y = tf.keras.utils.to_categorical(y)

pairs = attachLabels(x, y)
split = int(0.8 * len(pairs))
train, test = pairs[:split], pairs[split:]

# saveData(train, test, False)
# saveData(train, test, True)
# print()

hashParser = pf.HashParser()
mergeHashTree, mergeLabels = hashParser.parse(merge)
quickHashTree, quickLabels = hashParser.parse(quick)

x_hash = mergeHashTree + quickHashTree 
y_hash = mergeLabels + quickLabels 
y_hash = tf.keras.utils.to_categorical(y_hash)
pairs = attachLabels(x_hash, y_hash)
split = int(0.8 * len(pairs))
train, test = pairs[:split], pairs[split:]

# saveHashData(train, test, True)
# saveHashData(train, test, False)
# print()


