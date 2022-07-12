import numpy as np
from ParsingAndEmbeddingLayers.Text.TextParser import TextParser
from Networks.MLP import MLP
from Networks.RNN import RNN
from Networks.NaiveBayes import NBClassifier
from Networks.DenseModel import runDenseModel
from Networks.SKLearnClassifiers import SGDClassify, rfClassify, SVMClassify

tp = TextParser()
x_train, y_train, x_test, y_test = tp.getVectorizedTextData()
lstm = "lstm"
gru = "gru"
simpleRNN = "rnn"

def runMLPModels():
    layers = [len(x_train[0]), 128, 128, 2]
    epochs = 10
    lr = 0.001
    mlp1 = MLP(x_train, y_train, layers, "relu", lr, epochs)
    metrics1 = mlp1.runFFModel(x_train, y_train, x_test, y_test)

    mlp2 = MLP(x_train, y_train, layers, "tanh", lr, epochs)
    metrics2 = mlp2.runFFModel(x_train, y_train, x_test, y_test)

    mlp3 = MLP(x_train, y_train, layers, "softmax", lr, epochs)
    metrics3 = mlp3.runFFModel(x_train, y_train, x_test, y_test)

    mlp4 = MLP(x_train, y_train, layers, "sigmoid", lr, epochs)
    metrics4 = mlp4.runFFModel(x_train, y_train, x_test, y_test)

    print("USING THE MULTI-LAYER PERCEPTRON")
    print("USING RELU")
    print("Average loss:", np.average(metrics1['trainingLoss']), "Average training accuracy:", 
    np.average(metrics1['trainingAccuracy']), "Average validation accuracy:", 
    np.average(metrics1['validationAccuracy']), "\n") 

    print("USING TANH")
    print("Average loss:", np.average(metrics2['trainingLoss']), "Average training accuracy:", 
    np.average(metrics2['trainingAccuracy']), "Average validation accuracy:", 
    np.average(metrics2['validationAccuracy']), "\n") 

    print("USING SOFTMAX")
    print("Average loss:", np.average(metrics3['trainingLoss']), "Average training accuracy:", 
    np.average(metrics3['trainingAccuracy']), "Average validation accuracy:", 
    np.average(metrics3['validationAccuracy']), "\n") 

    print("USING SIGMOID")
    print("Average loss:", np.average(metrics4['trainingLoss']), "Average training accuracy:", 
    np.average(metrics4['trainingAccuracy']), "Average validation accuracy:", 
    np.average(metrics4['validationAccuracy']), "\n") 

def runReluRNNs():
    reluModel = RNN("lstm", x_train, y_train, x_test, y_test, "relu")

    reluModel.runModel(lstm, 256, 10, 10)
    reluModel.runModel(gru, 256, 10, 10)
    reluModel.runModel(simpleRNN, 256, 10, 10)

def runSoftmaxRNNs():
    softmaxModel = RNN("lstm", x_train, y_train, x_test, y_test, "softmax")

    softmaxModel.runModel(lstm, 256, 10, 10)
    softmaxModel.runModel(gru, 256, 10, 10)
    softmaxModel.runModel(simpleRNN, 256, 10, 10)

def runTanhRNNs():
    tanhModel = RNN("lstm", x_train, y_train, x_test, y_test, "tanh")

    tanhModel.runModel(lstm, 256, 10, 10)
    tanhModel.runModel(gru, 256, 10, 10)
    tanhModel.runModel(simpleRNN, 256, 10, 10)

def runSigmoidRNNs():
    sigmoidModel = RNN("lstm", x_train, y_train, x_test, y_test, "sigmoid")

    sigmoidModel.runModel(lstm, 256, 10, 10) 
    sigmoidModel.runModel(gru, 256, 10, 10)
    sigmoidModel.runModel(simpleRNN, 256, 10, 10)

def runDenseTextModels():
    print("With RELU")
    runDenseModel(x_train, y_train, x_test, y_test, "relu", 20, 10)

    print("With SOFTMAX")    
    runDenseModel(x_train, y_train, x_test, y_test, "softmax", 20, 10)

    print("With TANH")    
    runDenseModel(x_train, y_train, x_test, y_test, "tanh", 20, 10)

    print("With SIGMOID")    
    runDenseModel(x_train, y_train, x_test, y_test, "sigmoid", 20, 10)
   
def runGaussianNBC():
    # convert the tensors to simple Python lists
    x, y = [], []
    for i in x_train:
        x.append(list(i.numpy()))
    for i in x_test:
        x.append(list(i.numpy()))
    
    for i in y_train:
        i = list(i)
        label = i.index(1.0)
        y.append(int(label))
    for i in y_test:
        i = list(i)
        label = i.index(1.0)
        y.append(int(label))
    nbc = NBClassifier(x, y)
    print("Gaussian NB Classifier Accuracy:", nbc.gaussianCrossValidation(x, y)) #86.875

def runSKLearnClassifiers():
    xTrain, yTrain, xTest, yTest = [], [], [], []
    for i in x_train:
        xTrain.append(list(i.numpy()))
    for i in x_test:
        xTest.append(list(i.numpy()))
    
    for i in y_train:
        i = list(i)
        label = i.index(1.0)
        yTrain.append(int(label))
    for i in y_test:
        i = list(i)
        label = i.index(1.0)
        yTest.append(int(label))

    print("SGD CLASSIFIER:", SGDClassify(xTrain, yTrain, xTest, yTest))
    print("RF CLASSIFIER:", rfClassify(xTrain, yTrain, xTest, yTest)) 
    print("SVM CLASSIFIER:", SVMClassify(xTrain, yTrain, xTest, yTest)) 


runMLPModels()
runReluRNNs()
runSoftmaxRNNs()
runTanhRNNs()
runSigmoidRNNs()
runDenseTextModels()
runGaussianNBC()
runSKLearnClassifiers()

