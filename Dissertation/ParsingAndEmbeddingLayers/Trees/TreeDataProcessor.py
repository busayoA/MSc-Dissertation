import random
import tensorflow as tf
from ParsingAndEmbeddingLayers.Trees.TreeEmbeddingLayer import TreeEmbeddingLayer
from ParsingAndEmbeddingLayers.Trees.TreeParser import TreeParser
from os.path import dirname, join

merge = "./Datasets/Merge Sort"
quick = "./Datasets/Quick Sort"

currentDirectory = dirname(__file__) #the current working directory on the device
pathSplit = "/ParsingAndEmbeddingLayers"
head = currentDirectory.split(pathSplit) #split the path into two separate parts
path = head[0] 

merge = join(path, merge) #join the directory path to the absolute path
quick = join(path, quick)

def attachLabels(x, y):
    """
    Pair up the class labels with the actual data and randomly shuffle the result

    x: The list containing the actual data
    y: The class labels

    Returns 
    pairs - A list containing data and the labels in tuples
    """
    pairs = []
    for index in range(len(x)):
        pairs.append([x[index], y[index]])
    random.shuffle(pairs)
    return pairs

def getFileNames(hashed: bool):
    """
    Get the names of the files to be read depending on whether we are working
    with hashed data or not

    hashed: bool - Whether or not we are working with hashed data

    Returns
    xTrain - The name of the file containing the training data
    yTrain - The name of the file containing the training data labels
    xTest - The name of the file containing the testing data
    yTest - The name of the file containing the testing data labels
    """
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
    """
    Save the unhashed/vectorized training and testing data into lists

    train - The training data
    test - The testing data
    """
    print("\nCollecting training data", end="......")
    x_train, y_train = [], []
    for i in range(len(train)):
        if i % 2 == 0:
            print(end=".")
        current = train[i]
        embedding = TreeEmbeddingLayer(current) #embed all the nodes in the current tree
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

    # write the contents of each list into the appropriate files with hashed = false
    writeToFiles(x_train, y_train, x_test, y_test, False)
    
def saveHashData(train, test):
    """
    Save the hashed training and testing data into lists

    train - The hashed training data
    test - The hashed testing data
    """
    print("\nCollecting training data", end="......")
    x_train, y_train = [], []
    for i in range(len(train)):
        if i % 5 == 0:
            print(end=".")
        current = train[i]
        tree = current[0].getTreeEmbeddings(current[0]) #get the embedding of each node in the tree
        x_train.append(tree)
        y_train.append(train[i][1])

    # Repeat for the testing data
    print("\nCollecting testing data", end="......")
    x_test, y_test = [], []
    for i in range(len(test)):
        if i % 5 == 0:
            print(end=".")
        current = test[i]
        tree = current[0].getTreeEmbeddings(current[0])
        x_test.append(tree)
        y_test.append(test[i][1])

    # write the contents of the training and testing data into files with hashed = true
    writeToFiles(x_train, y_train, x_test, y_test, True)

def writeToFiles(x_train, y_train, x_test, y_test, hashed):
    """
    Write the contents of the inputs to this method into appropriate files

    x_train - The training data
    y_train - The training data labels
    x_test - The testing data
    y_test - The testing data labels
    hashed - Whether or not hashed is true
    """
    #collect the right file names based on the value of hashed
    xTrain, yTrain, xTest, yTest = getFileNames(hashed) 

    # Write to each file
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
    """
    Read the contents of a file containing either training or testing data

    filePath - The path to the file containing the appropriate embeddings
    
    Returns
    values - The formatted contents of 'filePath'
    """
    with open(filePath, 'r') as reader:
        values = reader.readlines()
    
    for x in range(len(values)):
        # remove all unnecessary characters
        values[x] = values[x].replace("[", "").replace("]", "").strip("\n")
        values[x] = values[x].split(",")
        values[x] = [float(i) for i in values[x]]

    return values

def readYFiles(filePath):
    """
    Read the contents of a file containing either training labels or testing data labels

    filePath - The path to the file containing the appropriate embeddings
    
    Returns
    values - The formatted contents of 'filePath'
    """
    with open(filePath, 'r') as reader:
        values = reader.readlines()
    
    for y in range(len(values)):
        # strip newlines and brackets and convert back to float
        values[y] = values[y].replace("[", "").replace("]", "").strip("\n")
        values[y] = values[y].split(" ")
        values[y] = [float(i) for i in values[y]]

    return values

def getData(hashed: bool):
    """
    This method is called by external classes when they want to access the contents of the saved files
    It is also called by external methods when running the final experiments

    hashed: bool - Whether or not hashed is true

    Returns
    x_train - The formatted contents of the training data file
    y_train - The formatted contents of the training data labels file
    x_test - The formatted contents of the testing data file
    y_test - The formatted contents of the testing data labels file
    """
    saveHashedFiles()
    print()
    saveUnhashedFiles()

    xTrain, yTrain, xTest, yTest = getFileNames(hashed)
    
    x_train, y_train, x_test, y_test = [], [], [], []
    x_train = readXFiles(xTrain)
    y_train = readYFiles(yTrain)

    x_test = readXFiles(xTest)
    y_test = readYFiles(yTest)

    return x_train, y_train, x_test, y_test

def saveUnhashedFiles():
    """
    Call the saveData method on the unhashed/vectorized files
    """
    parser = TreeParser(False)
    mergeTree, mergeLabels = parser.parse(merge)
    quickTree, quickLabels = parser.parse(quick)

    x = mergeTree + quickTree 
    y = mergeLabels + quickLabels 

    pairs = attachLabels(x, y) #attach class labels
    split = int(0.8 * len(pairs)) #split 80-20 for training and testing
    train, test = pairs[:split], pairs[split:]
    saveData(train, test)

def saveHashedFiles():
    """
    Call the saveHashData method on the hashed files
    """
    hashParser = TreeParser(True)
    mergeHashTree, mergeLabels = hashParser.parse(merge)
    quickHashTree, quickLabels = hashParser.parse(quick)

    x_hash = mergeHashTree + quickHashTree 
    y_hash = mergeLabels + quickLabels 

    hashedPairs = attachLabels(x_hash, y_hash) #attach class labels
    split_hash = int(0.8 * len(hashedPairs))
    train_hash, test_hash = hashedPairs[:split_hash], hashedPairs[split_hash:]
    saveHashData(train_hash, test_hash)

def tensorToList(xValues: tf.Tensor):
    """
    Convert a tensor to a list for processing by the Non-Deep Learning models

    xValues: tf.Tensor - The data values to be converted into a list
    """
    x = []
    for i in xValues:
        x.append(list(i.numpy()))
    
    return x

def floatToInt(y):
    """
    Convert a tensor to a list and a float to an int for processing by the Non-Deep Learning models

    y - The data values to be converted into a list of integers
    """
    y = list(y)
    for i in range(len(y)):
        j = y[i]
        j = list(j)
        j = j[0]
        y[i] = int(j)
    return y

