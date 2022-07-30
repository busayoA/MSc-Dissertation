import numpy as np
from Networks.MLP import MLP
from Networks.RNN import RNN
from Networks.NaiveBayes import NBClassifier
from Networks.DenseModel import runDenseModel
from Networks.SKLearnClassifiers import SGDClassify, rfClassify, SVMClassify
from ParsingAndEmbeddingLayers.Trees.TreeSegmentationLayer import TreeSegmentationLayer
from ParsingAndEmbeddingLayers.Trees import TreeDataProcessor as tdp
from ParsingAndEmbeddingLayers.Trees import TreeSegmentation as seg


# hashed = True
hashed = False
x_train_usum, x_train_umean, x_train_umax, x_train_umin, x_train_uprod, y_train = seg.getUnsortedSegmentTrainData(hashed)
x_test_usum, x_test_umean, x_test_umax, x_test_umin, x_test_uprod, y_test = seg.getUnsortedSegmentTestData(hashed)

lstm = "lstm"
gru = "gru"
simpleRNN = "rnn"

print("USING HASHED =", str(hashed).upper(), "DATA")
segmentCount = 40
segmentationLayer = TreeSegmentationLayer()
layers = [segmentCount, 128, 128, 2]
epochs = 10
lr = 0.001

def runUnsortedMLPModel(activationFunction: str):
    """
    Run the MLP on the tree data

    activationFunction: str - The activation function to apply
    """
    print("RUNNING RNN MODELS USING UNSORTED SEGMENTATION")
    model = MLP(x_train_umean, y_train, layers, activationFunction, lr, epochs)
    metrics = model.runFFModel(x_train_umean, y_train, x_test_umean, y_test)

    print("USING", activationFunction.upper())
    print("Loss:", np.average(metrics['trainingLoss']), "Training Accuracy:",
        np.average(metrics['trainingAccuracy']), "Validation accuracy:",
        np.average(metrics['validationAccuracy']), "\n")

def runLSTM(activationFunction: str):
    """
    Run the LSTM on the tree data

    activationFunction: str - The activation function to apply
    """
    model = RNN("lstm", x_train_umean, y_train, x_test_umean, y_test, activationFunction)
    model.runModel(lstm, 12, 10, 70)

def runGRU(activationFunction: str):
    """
    Run the GRU on the tree data

    activationFunction: str - The activation function to apply
    """
    model = RNN("gru", x_train_umean, y_train, x_test_umean, y_test, activationFunction)
    model.runModel(gru, 64, 20, 64)

def runSRNN(activationFunction: str):
    """
    Run the Simple RNN on the tree data

    activationFunction: str - The activation function to apply
    """
    model = RNN("rnn", x_train_umean, y_train, x_test_umean, y_test, activationFunction)
    model.runModel(simpleRNN, 64, 10, 64)

def runUnsortedDenseModel():
    """
    Run the Densely connnected model on the tree data using all 4 activation functions
    """
    print("DENSE UNSORTED MODEL AND SOFTMAX")
    runDenseModel(x_train_umean, y_train, x_test_umean, y_test, "softmax", 5, 10, "denseSegmented.hdf5")
    print("DENSE UNSORTED MODEL AND RELU")
    runDenseModel(x_train_umean, y_train, x_test_umean, y_test, "relu", 5, 10, "denseSegmented.hdf5")
    print("DENSE UNSORTED MODEL AND TANH")
    runDenseModel(x_train_umean, y_train, x_test_umean, y_test, "tanh", 5, 10, "denseSegmented.hdf5")
    print("DENSE UNSORTED MODEL AND SIGMOID")
    runDenseModel(x_train_umean, y_train, x_test_umean, y_test, "sigmoid", 5, 10, "denseSegmented.hdf5")

def runGaussianNBCUnsorted():
    """
    Run the Gaussian Naïve Bayes CLassifier on the tree data
    """
    # convert the training and testing data and their labels into lists from tensors
    x_train = tdp.tensorToList(x_train_umean)
    x_test = tdp.tensorToList(x_test_umean)

    yTrain = tdp.floatToInt(y_train)
    yTest = tdp.floatToInt(y_test)

    x, y = [], []
    for i in x_train:
        x.append(i)
    for i in x_test:
        x.append(i)
    
    for i in yTrain:
        y.append(i)
    for i in yTest:
        y.append(i)

    nbc = NBClassifier(x, y)
    print("Gaussian NB Classifier Accuracy:", nbc.gaussianCrossValidation(x, y))

def runSKLearnClassifiersUnsorted():
    """
    Run the SVM, SGD and RF classifiers on the tree data
    """
    yTrain = tdp.floatToInt(y_train)
    yTest = tdp.floatToInt(y_test)

    sgdUSumAccuracy = SGDClassify(x_train_umean, yTrain, x_test_umean, yTest)
    print("SGD CLASSIFIER AND UNSORTED:", sgdUSumAccuracy)

    rfUSumAccuracy = rfClassify(x_train_umean, yTrain, x_test_umean, yTest)
    print("RANDOM FOREST CLASSIFIER AND UNSORTED:", rfUSumAccuracy)

    svmUSumAccuracy = SVMClassify(x_train_umean, yTrain, x_test_umean, yTest)
    print("SVM CLASSIFIER AND UNSORTED:", svmUSumAccuracy)

# EXPERIMENTS AND RESULTS
runUnsortedMLPModel("relu")
runUnsortedMLPModel("tanh")
runUnsortedMLPModel("softmax")
runUnsortedMLPModel("sigmoid")

runLSTM("relu")
runLSTM("tanh")
runLSTM("softmax")
runLSTM("sigmoid")

runGRU("relu")
runGRU("tanh")
runGRU("softmax")
runGRU("sigmoid")

runSRNN("relu")
runSRNN("tanh")
runSRNN("softmax")
runSRNN("sigmoid")

runUnsortedDenseModel()

runGaussianNBCUnsorted()

runSKLearnClassifiersUnsorted()
