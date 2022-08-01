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
    """
    Run the Multilayer Perceptron Models on text
    """
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
    """
    Run the RNN Models using ReLu Activation 
    """
    reluModel = RNN("lstm", x_train, y_train, x_test, y_test, "relu")

    print("LSTM WITH RELU")
    reluModel.runModel(lstm, 256, 10, 10)

    print("GRU WITH RELU")
    reluModel.runModel(gru, 256, 10, 10)

    print("SRNN WITH RELU")
    reluModel.runModel(simpleRNN, 256, 10, 10)

def runSoftmaxRNNs():
    """
    Run the RNN Models using SoftMax Activation 
    """
    softmaxModel = RNN("lstm", x_train, y_train, x_test, y_test, "softmax")

    print("LSTM WITH SOFTMAX")
    softmaxModel.runModel(lstm, 256, 10, 10)

    print("GRY WITH SOFTMAX")
    softmaxModel.runModel(gru, 256, 10, 10)

    print("SRNN WITH SOFTMAX")
    softmaxModel.runModel(simpleRNN, 256, 10, 10)

def runTanhRNNs():
    """
    Run the RNN Models using Tanh Activation 
    """
    tanhModel = RNN("lstm", x_train, y_train, x_test, y_test, "tanh")

    print("LSTM WITH TANH")
    tanhModel.runModel(lstm, 256, 10, 10)

    print("GRU WITH TANH")
    tanhModel.runModel(gru, 256, 10, 10)

    print("SRNN WITH TANH")
    tanhModel.runModel(simpleRNN, 256, 10, 10)

def runSigmoidRNNs():
    """
    Run the RNN Models using Sigmoid Activation 
    """
    sigmoidModel = RNN("lstm", x_train, y_train, x_test, y_test, "sigmoid")

    print("LSTM WITH SIGMOID")
    sigmoidModel.runModel(lstm, 256, 10, 10) 

    print("GRU WITH SIGMOID")
    sigmoidModel.runModel(gru, 256, 10, 10)

    print("SRNN WITH SIGMOID")
    sigmoidModel.runModel(simpleRNN, 256, 10, 10)

def runDenseTextModels():
    """
    Run the densely connected models with the four different activation functiond
    """
    print("DENSE WITH RELU")
    runDenseModel(x_train, y_train, x_test, y_test, "relu", 20, 10)

    print("DENSE WITH  SOFTMAX")    
    runDenseModel(x_train, y_train, x_test, y_test, "softmax", 20, 10)

    print("DENSE WITH  TANH")    
    runDenseModel(x_train, y_train, x_test, y_test, "tanh", 20, 10)

    print("DENSE WITH  SIGMOID")    
    runDenseModel(x_train, y_train, x_test, y_test, "sigmoid", 20, 10)
   
def runGaussianNBC():
    """
    Run the Gaussian Naïve Bayes classifier on the text-based inputs
    """
    # convert the tensors to simple Python lists
    x, y = [], []
    for i in x_train:
        x.append(list(i.numpy()))
    for i in x_test:
        x.append(list(i.numpy()))
    
    # convert the y tensors into simple Python lists
    for i in y_train:
        i = list(i)
        label = i.index(1.0)
        y.append(int(label))
    for i in y_test:
        i = list(i)
        label = i.index(1.0)
        y.append(int(label))

    # Run the classifier on the converted lists
    nbc = NBClassifier(x, y)
    print("Gaussian NB Classifier Accuracy:", nbc.gaussianCrossValidation(x, y)) #86.875

def runSKLearnClassifiers():
    """
    Run the SGD, SVM and RF classifiers on the tex-based input
    """
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

