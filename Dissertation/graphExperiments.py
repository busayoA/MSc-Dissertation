import numpy as np
from Networks.MLP import MLP
from Networks.RNN import RNN
from Networks.NaiveBayes import NBClassifier
from Networks.DenseModel import runDenseModel
from Networks.SKLearnClassifiers import SGDClassify, rfClassify, SVMClassify
import ParsingAndEmbeddingLayers.Graphs.GraphDataProcessor as GDP

hashed = True  # if you want to test with hashed graphs, set HASHED to True
# hashed = False # else, set to False
gdp = GDP.GraphDataProcessor(hashed)

def runMLPonPaddedGraphs():
      """RUNNING ON PADDED GRAPHS"""
      x_train, y_train, x_test, y_test = gdp.runProcessor1()

      layers = [len(x_train[0]), 128, 128, 2]
      epochs = 10
      lr = 0.001
      mlp1 = MLP(x_train, y_train, layers, "relu", lr, epochs)
      metrics1 = mlp1.runFFModel(x_train, y_train, x_test, y_test)

      mlp2 = MLP(x_train, y_train, layers, "tanh", lr, epochs)
      metrics2 = mlp2.runFFModel(x_train, y_train, x_test, y_test)

      mlp3 = MLP(x_train, y_train, layers, "softmax", lr, epochs)
      metrics3 = mlp3.runFFModel(x_train, y_train, x_test, y_test)

      mlp4 = MLP(x_train, y_train, layers, "logsigmoid", lr, epochs)
      metrics4 = mlp4.runFFModel(x_train, y_train, x_test, y_test)

      print("USING THE MULTI-LAYER PERCEPTRON AND PADDED GRAPHS")
      print("USING RELU")
      print("Loss:", np.average(metrics1['trainingLoss']), "Training Accuracy:",
            np.average(metrics1['trainingAccuracy']), "Validation accuracy:",
            np.average(metrics1['validationAccuracy']), "\n")

      print("USING TANH")
      print("Loss:", np.average(metrics2['trainingLoss']), "Training Accuracy:",
            np.average(metrics2['trainingAccuracy']), "Validation accuracy:",
            np.average(metrics2['validationAccuracy']), "\n")

      print("USING SOFTMAX")
      print("Loss:", np.average(metrics3['trainingLoss']), "Training Accuracy:",
            np.average(metrics3['trainingAccuracy']), "Validation accuracy:",
            np.average(metrics3['validationAccuracy']), "\n")

      print("USING SIGMOID")
      print("Loss:", np.average(metrics4['trainingLoss']), "Training Accuracy:",
            np.average(metrics4['trainingAccuracy']), "Validation accuracy:",
            np.average(metrics4['validationAccuracy']), "\n")

def runMLPonSegmentedGraphs():
      """RUNNING ON SEGMENTED GRAPHS"""
      x_train, y_train, x_test, y_test = gdp.runProcessor3()

      layers = [len(x_train[0]), 128, 128, 2]
      epochs = 10
      lr = 0.001
      mlp1 = MLP(x_train, y_train, layers, "relu", lr, epochs)
      metrics1 = mlp1.runFFModel(x_train, y_train, x_test, y_test)

      mlp2 = MLP(x_train, y_train, layers, "tanh", lr, epochs)
      metrics2 = mlp2.runFFModel(x_train, y_train, x_test, y_test)

      mlp3 = MLP(x_train, y_train, layers, "softmax", lr, epochs)
      metrics3 = mlp3.runFFModel(x_train, y_train, x_test, y_test)

      mlp4 = MLP(x_train, y_train, layers, "logsigmoid", lr, epochs)
      metrics4 = mlp4.runFFModel(x_train, y_train, x_test, y_test)


      print("USING THE MULTI-LAYER PERCEPTRON AND SEGMENTATION")
      print("USING RELU")
      print("Loss:", np.average(metrics1['trainingLoss']), "Training Accuracy:",
            np.average(metrics1['trainingAccuracy']), "Validation accuracy:",
            np.average(metrics1['validationAccuracy']), "\n")

      print("USING TANH")
      print("Loss:", np.average(metrics2['trainingLoss']), "Training Accuracy:",
            np.average(metrics2['trainingAccuracy']), "Validation accuracy:",
            np.average(metrics2['validationAccuracy']), "\n")

      print("USING SOFTMAX")
      print("Loss:", np.average(metrics3['trainingLoss']), "Training Accuracy:",
            np.average(metrics3['trainingAccuracy']), "Validation accuracy:",
            np.average(metrics3['validationAccuracy']), "\n")

      print("USING SIGMOID")
      print("Loss:", np.average(metrics4['trainingLoss']), "Training Accuracy:",
            np.average(metrics4['trainingAccuracy']), "Validation accuracy:",
            np.average(metrics4['validationAccuracy']), "\n")

def runDenseModelonPaddedGraphs():
      x_train, y_train, x_test, y_test = gdp.runProcessor1()

      print("DENSE PADDED MODEL AND SOFTMAX")
      runDenseModel(x_train, y_train, x_test, y_test, "softmax", 5, 10, "densePaddedSoftmax.hdf5")
      print("DENSE PADDED MODEL AND RELU")
      runDenseModel(x_train, y_train, x_test, y_test, "relu", 5, 10, "densePaddedRelu.hdf5")
      print("DENSE PADDED MODEL AND TANH")
      runDenseModel(x_train, y_train, x_test, y_test, "tanh", 5, 10, "densePaddedTanh.hdf5")
      print("DENSE PADDED MODEL AND SIGMOID")
      runDenseModel(x_train, y_train, x_test, y_test, "sigmoid", 5, 10, "densePaddedSigmoid.hdf5")

def runDenseModelonSegmentedGraphs():
      x_train, y_train, x_test, y_test = gdp.runProcessor3()

      print("DENSE SEGMENTED MODEL AND SOFTMAX")
      runDenseModel(x_train, y_train, x_test, y_test, "softmax", 5, 10, "denseSegmented.hdf5")
      print("DENSE SEGMENTED MODEL AND RELU")
      runDenseModel(x_train, y_train, x_test, y_test, "relu", 5, 10, "denseSegmented.hdf5")
      print("DENSE SEGMENTED MODEL AND TANH")
      runDenseModel(x_train, y_train, x_test, y_test, "tanh", 5, 10, "denseSegmented.hdf5")
      print("DENSE SEGMENTED MODEL AND SIGMOID")
      runDenseModel(x_train, y_train, x_test, y_test, "sigmoid", 5, 10, "denseSegmented.hdf5")

def runGaussianNBConPaddedGraphs():
      x_train, y_train, x_test, y_test = gdp.runProcessor2()
      x, y = [], []
      for i in x_train:
            x.append(i)
      for i in x_test:
            x.append(i)
      
      for i in y_train:
            y.append(i)
      for i in y_test:
            y.append(i)
      nbc = NBClassifier(x, y)
      print("Gaussian NB CLassifier Accuracy:", nbc.gaussianCrossValidation(x, y))

def runGaussianNBConSegmentedGraphs():
      x_train, y_train, x_test, y_test = gdp.runProcessor4()
      x, y = [], []
      for i in x_train:
            x.append(i)
      for i in x_test:
            x.append(i)
      
      for i in y_train:
            y.append(i)
      for i in y_test:
            y.append(i)
      nbc = NBClassifier(x, y)
      print("Gaussian NB CLassifier Accuracy:", nbc.gaussianCrossValidation(x, y))

def runLSTMonPaddedGraphs(activationFunction: str):
      """RUNNING LSTM ON PADDED GRAPHS"""
      x_train, y_train, x_test, y_test = gdp.runProcessor1()
      print(activationFunction.upper())
      graphLSTM = "lstm"
      lstmModel = RNN(graphLSTM, x_train, y_train, x_test, y_test, activationFunction)
      lstmModel.runModel(graphLSTM, 256, 10, 5, "graphLSTMPadded.hdf5")

def runLSTMonSegmentedGraphs(activationFunction: str):
      """RUNNING LSTM ON PADDED GRAPHS"""
      x_train, y_train, x_test, y_test = gdp.runProcessor3()
      print(activationFunction.upper())
      graphLSTM = "lstm"
      lstmModel = RNN(graphLSTM, x_train, y_train, x_test, y_test, activationFunction)
      lstmModel.runModel(graphLSTM, 256, 10, 5, "graphLSTMSegmented.hdf5")

def runGRUonPaddedGraphs(activationFunction: str):
      """RUNNING GRU ON PADDED GRAPHS"""
      x_train, y_train, x_test, y_test = gdp.runProcessor1()
      print(activationFunction.upper())
      gru = "gru"
      gruModel = RNN(gru, x_train, y_train, x_test, y_test, activationFunction)
      gruModel.runModel(gru, 256, 10, 5, "graphGRUPadded.hdf5")

def runGRUonSegmentedGraphs(activationFunction: str):
      """RUNNING GRU ON PADDED GRAPHS"""
      x_train, y_train, x_test, y_test = gdp.runProcessor3()
      print(activationFunction.upper())
      gru = "gru"
      gruModel = RNN(gru, x_train, y_train, x_test, y_test, activationFunction)
      gruModel.runModel(gru, 256, 10, 5, "graphGRUSegmented.hdf5")

def runSRNNonPaddedGraphs(activationFunction: str):
      """RUNNING GRU ON PADDED GRAPHS"""
      x_train, y_train, x_test, y_test = gdp.runProcessor1()
      print(activationFunction.upper())
      srnn = "rnn"
      srnnModel = RNN(srnn, x_train, y_train, x_test, y_test, activationFunction)
      srnnModel.runModel(srnn, 256, 10, 5, "graphSRNNPadded.hdf5")

def runSRNNonSegmentedGraphs(activationFunction: str):
      """RUNNING GRU ON PADDED GRAPHS"""
      x_train, y_train, x_test, y_test = gdp.runProcessor3()
      print(activationFunction.upper())
      srnn = "rnn"
      srnnModel = RNN(srnn, x_train, y_train, x_test, y_test, activationFunction)
      srnnModel.runModel(srnn, 256, 10, 5, "graphSRNNSegmented.hdf5")

def runSKLearnClassifiersOnPaddedGraphs():
      """RUNNING ON PADDED GRAPHS"""
      x_train, y_train, x_test, y_test = gdp.runProcessor2()

      sgdUSumPadAccuracy = SGDClassify(x_train, y_train, x_test, y_test)
      print("SGD CLASSIFIER AND PADDED GRAPHS:", sgdUSumPadAccuracy)

      rfUSumPadAccuracy = rfClassify(x_train, y_train, x_test, y_test)
      print("RANDOM FOREST CLASSIFIER AND PADDED GRAPHS:", rfUSumPadAccuracy)

      svmUSumPadAccuracy = SVMClassify(x_train, y_train, x_test, y_test)
      print("SVM CLASSIFIER AND PADDED GRAPHS:", svmUSumPadAccuracy)

def runSKLearnClassifiersOnSegmentedGraphs():
      """RUNNING ON SEGMENTED GRAPHS"""
      x_train, y_train, x_test, y_test = gdp.runProcessor4()

      sgdUSumSegAccuracy = SGDClassify(x_train, y_train, x_test, y_test)
      print("SGD CLASSIFIER AND SEGMENTED GRAPHS:", sgdUSumSegAccuracy)

      rfUSumSegAccuracy = rfClassify(x_train, y_train, x_test, y_test)
      print("RANDOM FOREST CLASSIFIER AND SEGMENTED GRAPHS:", rfUSumSegAccuracy)

      svmUSumSegAccuracy = SVMClassify(x_train, y_train, x_test, y_test)
      print("SVM CLASSIFIER AND SEGMENTED GRAPHS:", svmUSumSegAccuracy)

runMLPonPaddedGraphs() 
runMLPonSegmentedGraphs()

runDenseModelonPaddedGraphs() 
runDenseModelonSegmentedGraphs()

runGaussianNBConPaddedGraphs()      
runGaussianNBConSegmentedGraphs() 

runLSTMonPaddedGraphs("relu") 
runLSTMonPaddedGraphs("tanh") 
runLSTMonPaddedGraphs("sigmoid")
runLSTMonPaddedGraphs("softmax") 

runLSTMonSegmentedGraphs("relu") 
runLSTMonSegmentedGraphs("tanh")
runLSTMonSegmentedGraphs("sigmoid")
runLSTMonSegmentedGraphs("softmax")

runGRUonPaddedGraphs("relu")  
runGRUonPaddedGraphs("tanh")
runGRUonPaddedGraphs("sigmoid")
runGRUonPaddedGraphs("softmax") 

runGRUonSegmentedGraphs("relu") 
runGRUonSegmentedGraphs("tanh") 
runGRUonSegmentedGraphs("sigmoid") 
runGRUonSegmentedGraphs("softmax") 

runSRNNonPaddedGraphs("relu") 
runSRNNonPaddedGraphs("tanh")
runSRNNonPaddedGraphs("sigmoid") 
runSRNNonPaddedGraphs("softmax") 

runSRNNonSegmentedGraphs("relu") 
runSRNNonSegmentedGraphs("tanh")
runSRNNonSegmentedGraphs("sigmoid") 
runSRNNonSegmentedGraphs("softmax") 

runSKLearnClassifiersOnPaddedGraphs() 
runSKLearnClassifiersOnSegmentedGraphs()

