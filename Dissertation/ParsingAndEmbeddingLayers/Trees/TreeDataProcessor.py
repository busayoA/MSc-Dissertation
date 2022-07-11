import random
from ParsingAndEmbeddingLayers.Trees.TreeEmbeddingLayer import TreeEmbeddingLayer
from ParsingAndEmbeddingLayers.Trees.TreeParser import TreeParser
from os.path import dirname, join

current_dir = dirname(__file__)
merge = join(current_dir, "./Data/Merge Sort")
quick = join(current_dir, "./Data/Quick Sort")

def attachLabels(x, y):
    pairs = []
    for index in range(len(x)):
        pairs.append([x[index], y[index]])
    random.shuffle(pairs)
    return pairs

def getFileNames(hashed: bool):
    current_dir = dirname(__file__)
    if hashed is False:
        xTrain = join(current_dir, "./Tree Data/tree_x_train.txt")
        yTrain = join(current_dir, "./Tree Data/tree_y_train.txt")
        xTest = join(current_dir, "./Tree Data/tree_x_test.txt")
        yTest = join(current_dir, "./Tree Data/tree_y_test.txt")
    else:
        xTrain = join(current_dir, "./Tree Data/tree_x_train_hashed.txt")
        yTrain = join(current_dir, "./Tree Data/tree_y_train_hashed.txt")
        xTest = join(current_dir, "./Tree Data/tree_x_test_hashed.txt")
        yTest = join(current_dir, "./Tree Data/tree_y_test_hashed.txt")

    return xTrain, yTrain, xTest, yTest

def saveData(train, test):
    print("\nCollecting training data", end="......")
    x_train, y_train = [], []
    for i in range(len(train)):
        if i % 2 == 0:
            print(end=".")
        current = train[i]
        embedding = TreeEmbeddingLayer(current)
        x_train.append(embedding.vectors)
        y_train.append(embedding.label)

    print("\nCollecting testing data", end="......")
    x_test, y_test = [], []
    for i in range(len(test)):
        if i % 5 == 0:
            print(end=".")
        embedding = TreeEmbeddingLayer(test[i])
        x_test.append(embedding.vectors)
        y_test.append(embedding.label)

    writeToFiles(x_train, y_train, x_test, y_test, False)
    
def saveHashData(train, test):
    print("\nCollecting training data", end="......")
    x_train, y_train = [], []
    for i in range(len(train)):
        if i % 5 == 0:
            print(end=".")
        current = train[i]
        tree = current[0].getTreeEmbeddings(current[0])
        x_train.append(tree)
        y_train.append(train[i][1])

    print("\nCollecting testing data", end="......")
    x_test, y_test = [], []
    for i in range(len(test)):
        if i % 5 == 0:
            print(end=".")
        current = test[i]
        tree = current[0].getTreeEmbeddings(current[0])
        x_test.append(tree)
        y_test.append(test[i][1])

    writeToFiles(x_train, y_train, x_test, y_test, True)

def writeToFiles(x_train, y_train, x_test, y_test, hashed):
    xTrain, yTrain, xTest, yTest = getFileNames(hashed)
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

def getData(hashed: bool):
    xTrain, yTrain, xTest, yTest = getFileNames(hashed)
    
    x_train, y_train, x_test, y_test = [], [], [], []
    x_train = readXFiles(xTrain)
    y_train = readYFiles(yTrain)

    x_test = readXFiles(xTest)
    y_test = readYFiles(yTest)

    return x_train, y_train, x_test, y_test

# RUN THE DATA PROCESSOR ON THE UNHASHED TREES
def saveUnhashedFiles():
    parser = TreeParser(False)
    mergeTree, mergeLabels = parser.parse(merge)
    quickTree, quickLabels = parser.parse(quick)

    x = mergeTree + quickTree 
    y = mergeLabels + quickLabels 

    pairs = attachLabels(x, y)
    split = int(0.8 * len(pairs))
    train, test = pairs[:split], pairs[split:]
    saveData(train, test)

# RUN THE DATA PROCESSOR ON THE HASHED TREES
def saveHashedFiles():
    hashParser = TreeParser(True)
    mergeHashTree, mergeLabels = hashParser.parse(merge)
    quickHashTree, quickLabels = hashParser.parse(quick)

    x_hash = mergeHashTree + quickHashTree 
    y_hash = mergeLabels + quickLabels 
    hashedPairs = attachLabels(x_hash, y_hash)
    split_hash = int(0.8 * len(hashedPairs))
    train_hash, test_hash = hashedPairs[:split_hash], hashedPairs[split_hash:]
    saveHashData(train_hash, test_hash)

def tensorToList(xValues):
    x = []
    for i in xValues:
        x.append(list(i.numpy()))
    
    return x

def floatToInt(y):
    y = list(y)
    for i in range(len(y)):
        j = y[i]
        j = list(j)
        j = j[0]
        y[i] = int(j)
    return y

# saveHashedFiles()
# print()
# saveUnhashedFiles()
