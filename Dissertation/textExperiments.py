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
    # Average loss: 0.9099208 Average training accuracy: 0.5549549549549548 Average validation accuracy: 0.512 

    mlp2 = MLP(x_train, y_train, layers, "tanh", lr, epochs)
    metrics2 = mlp2.runFFModel(x_train, y_train, x_test, y_test)
    # Average loss: 0.82113314 Average training accuracy: 0.581981981981982 Average validation accuracy: 0.4800000000000001 

    mlp3 = MLP(x_train, y_train, layers, "softmax", lr, epochs)
    metrics3 = mlp3.runFFModel(x_train, y_train, x_test, y_test)
    # 0.78467464 Average training accuracy: 0.4774774774774775 Average validation accuracy: 0.4800000000000001 

    mlp4 = MLP(x_train, y_train, layers, "logsigmoid", lr, epochs)
    metrics4 = mlp4.runFFModel(x_train, y_train, x_test, y_test)
    # 0.77560914 Average training accuracy: 0.5225225225225225 Average validation accuracy: 0.5199999999999999

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
                # loss: 0.7036 - accuracy: 0.5135 - val_loss: 0.7006 - val_accuracy: 0.5200
    reluModel.runModel(gru, 256, 10, 10)
                # loss: 0.6954 - accuracy: 0.5225 - val_loss: 0.7001 - val_accuracy: 0.5200
    reluModel.runModel(simpleRNN, 256, 10, 10)
                # loss: 6.9475 - accuracy: 0.5045 - val_loss: 6.6739 - val_accuracy: 0.5200

def runSoftmaxRNNs():
    softmaxModel = RNN("lstm", x_train, y_train, x_test, y_test, "softmax")

    softmaxModel.runModel(lstm, 256, 10, 10)
                # loss: 0.6930 - accuracy: 0.5135 - val_loss: 0.6884 - val_accuracy: 0.5200
    softmaxModel.runModel(gru, 256, 10, 10)
                # loss: 0.6842 - accuracy: 0.5586 - val_loss: 0.6850 - val_accuracy: 0.6000
    softmaxModel.runModel(simpleRNN, 256, 10, 10)
                # loss: 0.9355 - accuracy: 0.4685 - val_loss: 0.7289 - val_accuracy: 0.5000

def runTanhRNNs():
    tanhModel = RNN("lstm", x_train, y_train, x_test, y_test, "tanh")

    tanhModel.runModel(lstm, 256, 10, 10)
                #loss: 0.7101 - accuracy: 0.5225 - val_loss: 0.7003 - val_accuracy: 0.5200
    tanhModel.runModel(gru, 256, 10, 10)
                # loss: 0.7121 - accuracy: 0.5045 - val_loss: 0.7060 - val_accuracy: 0.5200
    tanhModel.runModel(simpleRNN, 256, 10, 10)
                # loss: 2.6537 - accuracy: 0.4955 - val_loss: 1.2998 - val_accuracy: 0.4200

def runSigmoidRNNs():
    sigmoidModel = RNN("lstm", x_train, y_train, x_test, y_test, "sigmoid")

    sigmoidModel.runModel(lstm, 256, 10, 10) 
                    #loss: 0.7023 - accuracy: 0.5315 - val_loss: 0.6977 - val_accuracy: 0.5200
    sigmoidModel.runModel(gru, 256, 10, 10)
                    #loss: 0.6949 - accuracy: 0.5225 - val_loss: 0.6957 - val_accuracy: 0.5200
    sigmoidModel.runModel(simpleRNN, 256, 10, 10)
                    #loss: 6.5460 - accuracy: 0.5225 - val_loss: 7.3618 - val_accuracy: 0.5200


def runDenseTextModels():
    print("With RELU")
    runDenseModel(x_train, y_train, x_test, y_test, "relu", 20, 10)
    # loss: 0.6646 - accuracy: 0.8288 - val_loss: 0.7965 - val_accuracy: 0.6800

    print("With SOFTMAX")    
    runDenseModel(x_train, y_train, x_test, y_test, "softmax", 20, 10)
    #  loss: 0.6931 - accuracy: 0.5766 - val_loss: 0.6931 - val_accuracy: 0.5200

    print("With TANH")    
    runDenseModel(x_train, y_train, x_test, y_test, "tanh", 20, 10)
    # loss: 3.9075 - accuracy: 0.5315 - val_loss: 3.9766 - val_accuracy: 0.5200

    print("With SIGMOID")    
    runDenseModel(x_train, y_train, x_test, y_test, "sigmoid", 20, 10)
    # loss: 0.6890 - accuracy: 0.5225 - val_loss: 0.6905 - val_accuracy: 0.5200

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

    print("SGD CLASSIFIER:", SGDClassify(xTrain, yTrain, xTest, yTest)) #0.64
    print("RF CLASSIFIER:", rfClassify(xTrain, yTrain, xTest, yTest)) #1.0
    print("SVM CLASSIFIER:", SVMClassify(xTrain, yTrain, xTest, yTest)) #0.76


# runMLPModels()
# runReluRNNs()
# runSoftmaxRNNs()
# runTanhRNNs()
# runSigmoidRNNs()
# runDenseTextModels()
# runGaussianNBC()
# runSKLearnClassifiers()

