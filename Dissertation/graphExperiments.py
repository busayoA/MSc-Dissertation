import numpy as np
from Networks.MLP import MLP
from Networks.RNN import RNN
from Networks.NaiveBayes import NBClassifier
from Networks.DenseModel import runDenseModel
from Networks.SKLearnClassifiers import SGDClassify, rfClassify, SVMClassify
import ParsingAndEmbeddingLayers.Graphs.GraphDataProcessor as GDP

# hashed = True  # if you want to test with hashed graphs, set HASHED to True
hashed = False # else, set to False
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

# runMLPonPaddedGraphs()  # Loss: 9.403   TA: 0.5098  VA: 0.6224 - UNHASHED RELU
#                         # Loss: 0.8256  TA: 0.5625  VA: 0.6327 - UNHASHED TANH
#                         # Loss: 0.6944  TA: 0.4821  VA: 0.3673 - UNHASHED SOFTMAX
#                         # Loss: 0.7712  TA: 0.4911  VA: 0.5918 - UNHASHED SIGMOID

#                         # Loss:    TA:   VA:  - HASHED RELU
#                         # Loss:   TA:   VA:  - HASHED TANH
#                         # Loss:   TA:   VA:  - HASHED SOFTMAX
#                         # Loss:   TA:   VA:  - HASHED SIGMOID

# runMLPonSegmentedGraphs()# Loss: 2.3858   TA: 0.4848  VA: 0.6265 - UNHASHED RELU
#                         # Loss: 0.9311  TA: 0.4464  VA: 0.5102  - UNHASHED TANH
#                         # Loss: 0.6965  TA: 0.5089  VA: 0.4081 - UNHASHED SOFTMAX
#                         # Loss: 0.7171  TA: 0.5089  VA: 0.4081 - UNHASHED SIGMOID

#                         # Loss:    TA:   VA:  - HASHED RELU
#                         # Loss:   TA:   VA:  - HASHED TANH
#                         # Loss:   TA:   VA:  - HASHED SOFTMAX
#                         # Loss:   TA:   VA:  - HASHED SIGMOID

# runDenseModelonPaddedGraphs() # Loss: 0.0261, 2.0560  TA: 1.0  VA: 0.4286 - UNHASHED
#                               # Loss:  TA:   VA: - HASHED
# runDenseModelonSegmentedGraphs() # Loss:  TA:   VA: - UNHASHED
#                                  # Loss: 0.5153, 0.8751 TA: 75.89   VA: 0.5715 - HASHED

# runGaussianNBConPaddedGraphs() # Average Accuracy: 51.25%- UNHASHED
#                                     # Average Accuracy: - HASHED      
# runGaussianNBConSegmentedGraphs() # Average Accuracy: 51.25% - UNHASHED
#                                     # Average Accuracy: - HASHED  

runLSTMonPaddedGraphs("relu") # Loss:  TA:   VA: - UNHASHED
                              #    Loss:  TA:   VA: - HASHED
runLSTMonPaddedGraphs("tanh") # Loss:  TA:   VA: - UNHASHED
                              #    Loss:  TA:   VA: - HASHED
runLSTMonPaddedGraphs("sigmoid") # Loss:  TA:   VA: - UNHASHED
                              #    Loss:  TA:   VA: - HASHED
runLSTMonPaddedGraphs("softmax") # Loss:  TA:   VA: - UNHASHED
                              #    Loss:  TA:   VA: - HASHED

# runLSTMonSegmentedGraphs("relu")# Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED
# runLSTMonSegmentedGraphs("tanh") # Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED
# runLSTMonSegmentedGraphs("sigmoid") ## Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED
# runLSTMonSegmentedGraphs("softmax") # Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED

# runGRUonPaddedGraphs("relu")  # Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED
# runGRUonPaddedGraphs("tanh") # Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED
# runGRUonPaddedGraphs("sigmoid") # Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED
# runGRUonPaddedGraphs("softmax") # Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED

# runGRUonSegmentedGraphs("relu") # Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED
# runGRUonSegmentedGraphs("tanh") # Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED
# runGRUonSegmentedGraphs("sigmoid") # Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED
# runGRUonSegmentedGraphs("softmax") # Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED

# runSRNNonPaddedGraphs("relu") # Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED
# runSRNNonPaddedGraphs("tanh") # Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED
# runSRNNonPaddedGraphs("sigmoid") # Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED
# runSRNNonPaddedGraphs("softmax") # Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED

# runSRNNonSegmentedGraphs("relu") # Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED
# runSRNNonSegmentedGraphs("tanh")# Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED
# runSRNNonSegmentedGraphs("sigmoid") # Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED
# runSRNNonSegmentedGraphs("softmax") # Loss:  TA:   VA: - UNHASHED
                                 # Loss:  TA:   VA: - HASHED

# runSKLearnClassifiersOnPaddedGraphs() # SGD-48.98, RF-42.86, SVM-46.94 - UNHASHED, 
#                                           # SGD-46.94, RF-46.94, SVM-46.94 - HASHED
# runSKLearnClassifiersOnSegmentedGraphs() # SGD-53.06, RF-53.06, SVM-59.18 - UNHASHED
#                                           # SGD-46.94, RF-46.94, SVM-46.94 - HASHED

