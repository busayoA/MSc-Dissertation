import numpy as np
from Networks.MLP import MLP
from Networks.RNN import RNN
from Networks.NaiveBayes import NBClassifier
from Networks.DenseModel import runDenseModel
from Networks.SKLearnClassifiers import SGDClassify, rfClassify, SVMClassify
import ParsingAndEmbeddingLayers.Graphs.GraphDataProcessor as GDP

# hashed = True  # if you want to test with hashed graphs, set HASHED to True
hashed = True # else, set to False
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


      print("USING THE MULTI-LAYER PERCEPTRON ANF SEGMENTATION")
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

def runDenseModelonPaddedGraphs():
      x_train, y_train, x_test, y_test = gdp.runProcessor1()

      runDenseModel(x_train, y_train, x_test, y_test, "densePaddedHashed.hdf5")

def runDenseModelonSegmentedGraphs():
      x_train, y_train, x_test, y_test = gdp.runProcessor3()

      runDenseModel(x_train, y_train, x_test, y_test, "denseSegmentedHashed.hdf5")

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
      graphLSTM = "lstm"
      lstmModel = RNN(graphLSTM, x_train, y_train, x_test, y_test, activationFunction)
      lstmModel.runModel(graphLSTM, "graphLSTMPadded.hdf5", 256, 30)

def runLSTMonSegmentedGraphs(activationFunction: str):
      """RUNNING LSTM ON PADDED GRAPHS"""
      x_train, y_train, x_test, y_test = gdp.runProcessor3()
      graphLSTM = "lstm"
      lstmModel = RNN(graphLSTM, x_train, y_train, x_test, y_test, activationFunction)
      lstmModel.runModel(graphLSTM, "graphLSTMSegmented.hdf5", 256, 30)

def runGRUonPaddedGraphs(activationFunction: str):
      """RUNNING GRU ON PADDED GRAPHS"""
      x_train, y_train, x_test, y_test = gdp.runProcessor1()
      gru = "gru"
      gruModel = RNN(gru, x_train, y_train, x_test, y_test, activationFunction)
      gruModel.runModel(gru, "graphGRUPadded.hdf5", 256, 30)

def runGRUonSegmentedGraphs(activationFunction: str):
      """RUNNING GRU ON PADDED GRAPHS"""
      x_train, y_train, x_test, y_test = gdp.runProcessor3()
      gru = "gru"
      gruModel = RNN(gru, x_train, y_train, x_test, y_test, activationFunction)
      gruModel.runModel(gru, "graphGRUSegmented.hdf5", 256, 30)

def runSRNNonPaddedGraphs(activationFunction: str):
      """RUNNING GRU ON PADDED GRAPHS"""
      x_train, y_train, x_test, y_test = gdp.runProcessor1()
      srnn = "rnn"
      srnnModel = RNN(srnn, x_train, y_train, x_test, y_test, activationFunction)
      srnnModel.runModel(srnn, "graphSRNNPadded.hdf5", 256, 30)

def runSRNNonSegmentedGraphs(activationFunction: str):
      """RUNNING GRU ON PADDED GRAPHS"""
      x_train, y_train, x_test, y_test = gdp.runProcessor3()
      srnn = "rnn"
      srnnModel = RNN(srnn, x_train, y_train, x_test, y_test, activationFunction)
      srnnModel.runModel(srnn, "graphSRNNSegmented.hdf5", 256, 30)

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

# runMLPonPaddedGraphs() # ACCURACY: 60%-TANH, 56%-SOFTMAX, 55%-RELU, 48%-SIGMOID - UNHASHED
#                         # ACCURACY: 54.46%-TANH, 54.46%-SOFTMAX, 54.46%-RELU, 45.54%-SIGMOID - HASHED
# runMLPonSegmentedGraphs() # 62.5%-TANH, 50.89%-SOFTMAX, 50%-SIGMOID, 45.27%-RELU - UNHASHED
#                         # ACCURACY: 54.46%-TANH, 45.54%-SOFTMAX, 54.46%-RELU, 54.46%-SIGMOID - HASHED

# runDenseModelonPaddedGraphs() # 100% Training Accuracy (Indicates Overfitting) 45% Validation accuracy - UNHASHED
#                                     # % 54.46% Training Accuracy, 46.94% Validation Accuracy - HASHED
# runDenseModelonSegmentedGraphs() # 75% Training Accuracy, 61.22% Validation accuracy - UNHASHED
#                                     # % 54.46% Training Accuracy, 46.94% Validation Accuracy - HASHED

# runGaussianNBConPaddedGraphs() #50.625 - UNHASHED, 47.5% HASHED
# runGaussianNBConSegmentedGraphs() #54.375 - UNHASHED, 52.5% HASHED

# runLSTMonPaddedGraphs("relu") #51.79% Training Accuracy, 59.18% Validation Accuracy - UNHASHED
#                                     # 46.43% Training Accuracy, 53.06% Validation Accuracy - HASHED
# runLSTMonPaddedGraphs("tanh") #49.11% Training Accuracy, 59.18% Validation Accuracy - UNHASHED
#                                     # 55.36% Training Accuracy, 42.86% Validation Accuracy - HASHED
# runLSTMonPaddedGraphs("sigmoid") #49.11% Training Accuracy, 57.14% Validation Accuracy - UNHASHED
#                                     # 49.11% Training Accuracy, 42.86% Validation Accuracy - HASHED
# runLSTMonPaddedGraphs("softmax") #49.11% Training Accuracy, 59.18% Validation Accuracy - UNHASHED
#                                     # 51.79% Training Accuracy, 57.14% Validation Accuracy - HASHED

# runLSTMonSegmentedGraphs("relu") #55.36.% Training Accuracy, 53.06% Validation Accuracy - UNHASHED
#                                     # 43.75% Training Accuracy,46.94 % Validation Accuracy - HASHED
# runLSTMonSegmentedGraphs("tanh") #57.14% Training Accuracy, 46.94% Validation Accuracy - UNHASHED
#                                     # 47.32% Training Accuracy, 53.06% Validation Accuracy - HASHED
# runLSTMonSegmentedGraphs("sigmoid") #55.36% Training Accuracy, 48.98% Validation Accuracy - UNHASHED
#                                     # 41.07% Training Accuracy, 53.06% Validation Accuracy - HASHED
# runLSTMonSegmentedGraphs("softmax") #57.14% Training Accuracy, 48.98% Validation Accuracy - UNHASHED
#                                     # 46.43% Training Accuracy, 53.06% Validation Accuracy - HASHED

"""MAKE SURE YOU CHECK WHAT HASHED IS SET TO BEFORE YOU RUN THIS BUSAYO!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
runGRUonPaddedGraphs("relu")  #50.89.% Training Accuracy, 55.10% Validation Accuracy - UNHASHED
                                    # 28.57% Training Accuracy, 20.41% Validation Accuracy - HASHED
runGRUonPaddedGraphs("tanh") #50.00.% Training Accuracy, 61.22% Validation Accuracy - UNHASHED
                                    # 48.21% Training Accuracy, 46.94% Validation Accuracy - HASHED
runGRUonPaddedGraphs("sigmoid") #__.__.% Training Accuracy, __.__% Validation Accuracy - UNHASHED
                                    # 53.57% Training Accuracy, 44.90% Validation Accuracy - HASHED 
runGRUonPaddedGraphs("softmax") #__.__.% Training Accuracy, __.__% Validation Accuracy - UNHASHED
                                    # 33.93% Training Accuracy, 22.45% Validation Accuracy - HASHED

# runGRUonSegmentedGraphs("relu") #57.14.% Training Accuracy, 38.78_% Validation Accuracy - UNHASHED
#                                     # 44.64% Training Accuracy, 53.06% Validation Accuracy - HASHED
# runGRUonSegmentedGraphs("tanh") #54.46.% Training Accuracy, 48.98% Validation Accuracy - UNHASHED
#                                     # 51.79% Training Accuracy, 46.94% Validation Accuracy - HASHED
# runGRUonSegmentedGraphs("sigmoid") #56.25.% Training Accuracy, 42.86% Validation Accuracy - UNHASHED
#                                     # 49.11% Training Accuracy, 53.06% Validation Accuracy - HASHED
# runGRUonSegmentedGraphs("softmax") #58.93.% Training Accuracy, 42.86% Validation Accuracy - UNHASHED
#                                     # 44.64% Training Accuracy, 53.06% Validation Accuracy - HASHED

# runSRNNonPaddedGraphs("relu") #__.__.% Training Accuracy, __.__% Validation Accuracy - UNHASHED 
#                                     # % Training Accuracy, % Validation Accuracy - HASHED
# runSRNNonPaddedGraphs("tanh") #__.__.% Training Accuracy, __.__% Validation Accuracy - UNHASHED
#                                     # % Training Accuracy, % Validation Accuracy - HASHED
# runSRNNonPaddedGraphs("sigmoid") #__.__.% Training Accuracy, __.__% Validation Accuracy - UNHASHED
#                                     # % Training Accuracy, % Validation Accuracy - HASHED
# runSRNNonPaddedGraphs("softmax") #__.__.% Training Accuracy, __.__% Validation Accuracy - UNHASHED
#                                     # % Training Accuracy, % Validation Accuracy - HASHED

# runSRNNonSegmentedGraphs("relu") # 48.21% Training Accuracy, 42.86% Validation Accuracy - UNHASHED
# #                                     # 52.68% Training Accuracy, 42.86% Validation Accuracy - HASHED
# runSRNNonSegmentedGraphs("tanh") #50.98.% Training Accuracy, 40.82% Validation Accuracy - UNHASHED
# #                                     # 42.86% Training Accuracy, 53.06% Validation Accuracy - HASHED
# runSRNNonSegmentedGraphs("sigmoid") #50.89% Training Accuracy, 44.90% Validation Accuracy - UNHASHED
# #                                     # 54.46% Training Accuracy, 46.94% Validation Accuracy - HASHED
# runSRNNonSegmentedGraphs("softmax") #44.64% Training Accuracy, 59.18% Validation Accuracy - UNHASHED
# #                                     # 41.07% Training Accuracy, 44.9% Validation Accuracy - HASHED

# runSKLearnClassifiersOnPaddedGraphs() # SGD-48.98, RF-42.86, SVM-46.94 - UNHASHED, 
#                                           # SGD-46.94, RF-46.94, SVM-46.94 - HASHED
# runSKLearnClassifiersOnSegmentedGraphs() # SGD-53.06, RF-53.06, SVM-59.18 - UNHASHED
#                                           # SGD-46.94, RF-46.94, SVM-46.94 - HASHED

