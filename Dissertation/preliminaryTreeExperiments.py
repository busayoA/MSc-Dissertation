import numpy as np
from Networks.MLP import MLP
from Networks.RNN import RNN
from Networks.NaiveBayes import NBClassifier
from Networks.DenseModel import runDenseModel
from Networks.SKLearnClassifiers import SGDClassify, rfClassify, SVMClassify
from ParsingAndEmbeddingLayers.Trees.TreeSegmentationLayer import TreeSegmentationLayer
from ParsingAndEmbeddingLayers.Trees import TreeSegmentation as seg


hashed = True
# hashed = False
x_train_usum, x_train_umean, x_train_umax, x_train_umin, x_train_uprod, y_train = seg.getUnsortedSegmentTrainData(hashed)
x_test_usum, x_test_umean, x_test_umax, x_test_umin, x_test_uprod, y_test = seg.getUnsortedSegmentTestData(hashed)

x_train_sum, x_train_mean, x_train_max, x_train_min, x_train_prod, y_train = seg.getSortedSegmentTrainData(hashed)
x_test_sum, x_test_mean, x_test_max, x_test_min, x_test_prod, y_test = seg.getSortedSegmentTestData(hashed)

print("USING HASHED =", str(hashed).upper(), "DATA")
segmentCount = 40
segmentationLayer = TreeSegmentationLayer()
layers = [segmentCount, 64, 64, 2]
epochs = 30
lr = 0.05

# USING RELU ACTIVATION
print("RUNNING RNN MODELS USING UNSORTED SEGMENTATION AND HASHED NODES")
print("UNSORTED SEGMENT SUM AND RELU")
model1a = MLP(x_train_usum, y_train, layers, "relu", lr, epochs)
model1a.runFFModel(x_train_usum, y_train, x_test_usum, y_test)
print("UNSORTED SEGMENT MEAN AND RELU")
model1a.runFFModel(x_train_umean, y_train, x_test_umean, y_test)
print("UNSORTED SEGMENT MAX AND RELU")
model1a.runFFModel(x_train_umax, y_train, x_test_umax, y_test)
print("UNSORTED SEGMENT MIN AND RELU")
model1a.runFFModel(x_train_umin, y_train, x_test_umin, y_test)
print("UNSORTED SEGMENT PROD AND RELU")
model1a.runFFModel(x_train_uprod, y_train, x_test_uprod, y_test)
print()


# USING TANH ACTIVATION
print("UNSORTED SEGMENT SUM AND TANH")
model1b = MLP(x_train_usum, y_train, layers, "tanh", lr, epochs)
model1b.runFFModel(x_train_usum, y_train, x_test_usum, y_test)
print("UNSORTED SEGMENT MEAN AND TANH")
model1b.runFFModel(x_train_umean, y_train, x_test_umean, y_test)
print("UNSORTED SEGMENT MAX AND TANH")
model1b.runFFModel(x_train_umax, y_train, x_test_umax, y_test)
print("UNSORTED SEGMENT MIN AND TANH")
model1b.runFFModel(x_train_umin, y_train, x_test_umin, y_test)
print("UNSORTED SEGMENT PROD AND TANH")
model1b.runFFModel(x_train_uprod, y_train, x_test_uprod, y_test)
print()


# USING LOGSIGMOID ACTIVATION
print("UNSORTED SEGMENT SUM AND LOGSIGMOID")
model1c = MLP(x_train_usum, y_train, layers, "sigmoid", lr, epochs)
print("UNSORTED SEGMENT MEAN AND LOGSIGMOID")
model1c.runFFModel(x_train_umean, y_train, x_test_umean, y_test)
print("UNSORTED SEGMENT MAX AND LOGSIGMOID")
model1c.runFFModel(x_train_umax, y_train, x_test_umax, y_test)
print("UNSORTED SEGMENT MIN AND LOGSIGMOID")
model1c.runFFModel(x_train_umin, y_train, x_test_umin, y_test)
print("UNSORTED SEGMENT PROD AND LOGSIGMOID")
model1c.runFFModel(x_train_uprod, y_train, x_test_uprod, y_test)
print()

# USING SOFTMAX ACTIVATION
print("UNSORTED SEGMENT SUM AND SOFTMAX")
model1d = MLP(x_train_usum, y_train, layers, "softmax", lr, epochs)
model1d.runFFModel(x_train_usum, y_train, x_test_usum, y_test)
print("UNSORTED SEGMENT MEAN AND SOFTMAX")
model1d.runFFModel(x_train_umean, y_train, x_test_umean, y_test)
print("UNSORTED SEGMENT MAX AND SOFTMAX")
model1d.runFFModel(x_train_umax, y_train, x_test_umax, y_test)
print("UNSORTED SEGMENT MIN AND SOFTMAX")
model1d.runFFModel(x_train_umin, y_train, x_test_umin, y_test)
print("UNSORTED SEGMENT PROD AND SOFTMAX")
model1d.runFFModel(x_train_uprod, y_train, x_test_uprod, y_test)
print()



print("RUNNING RNN MODELS USING SORTED SEGMENTATION AND HASHED NODES")
# USING RELU ACTIVATION
print("SORTED SEGMENT SUM AND RELU")
model2a = MLP(x_train_usum, y_train, layers, "relu", lr, epochs)
model2a.runFFModel(x_train_sum, y_train, x_test_sum, y_test)
print("SORTED SEGMENT MEAN AND RELU")
model2a.runFFModel(x_train_mean, y_train, x_test_mean, y_test)
print("SORTED SEGMENT MAX AND RELU")
model2a.runFFModel(x_train_max, y_train, x_test_max, y_test)
print("SORTED SEGMENT MIN AND RELU")
model2a.runFFModel(x_train_min, y_train, x_test_min, y_test)
print("SORTED SEGMENT PROD AND RELU")
model2a.runFFModel(x_train_prod, y_train, x_test_prod, y_test)
print()


# USING TANH ACTIVATION
print("SORTED SEGMENT SUM AND TANH")
model2b = MLP(x_train_usum, y_train, layers, "tanh", lr, epochs)
model2b.runFFModel(x_train_sum, y_train, x_test_sum, y_test)
print("SORTED SEGMENT MEAN AND TANH")
model2b.runFFModel(x_train_mean, y_train, x_test_mean, y_test)
print("SORTED SEGMENT MAX AND TANH")
model2b.runFFModel(x_train_max, y_train, x_test_max, y_test)
print("SORTED SEGMENT MIN AND TANH")
model2b.runFFModel(x_train_min, y_train, x_test_min, y_test)
print("SORTED SEGMENT PROD AND TANH")
model2b.runFFModel(x_train_prod, y_train, x_test_prod, y_test)
print()


# USING LOGSIGMOID ACTIVATION
print("SORTED SEGMENT SUM AND LOGSIGMOID")
model2c = MLP(x_train_usum, y_train, layers, "sigmoid", lr, epochs)
model2c.runFFModel(x_train_sum, y_train, x_test_sum, y_test)
print("SORTED SEGMENT MEAN AND LOGSIGMOID")
model2c.runFFModel(x_train_mean, y_train, x_test_mean, y_test)
print("SORTED SEGMENT MAX AND LOGSIGMOID")
model2c.runFFModel(x_train_max, y_train, x_test_max, y_test)
print("SORTED SEGMENT MIN AND LOGSIGMOID")
model2c.runFFModel(x_train_min, y_train, x_test_min, y_test)
print("SORTED SEGMENT PROD AND LOGSIGMOID")
model2c.runFFModel(x_train_prod, y_train, x_test_prod, y_test)
print()

# USING SOFTMAX ACTIVATION
print("SORTED SEGMENT SUM AND SOFTMAX")
model2c = MLP(x_train_usum, y_train, layers, "softmax", lr, epochs)
model2c.runFFModel(x_train_sum, y_train, x_test_sum, y_test)
print("SORTED SEGMENT MEAN AND SOFTMAX")
model2c.runFFModel(x_train_mean, y_train, x_test_mean, y_test)
print("SORTED SEGMENT MAX AND SOFTMAX")
model2c.runFFModel(x_train_max, y_train, x_test_max, y_test)
print("SORTED SEGMENT MIN AND SOFTMAX")
model2c.runFFModel(x_train_min, y_train, x_test_min, y_test)
print("SORTED SEGMENT PROD AND SOFTMAX")
model2c.runFFModel(x_train_prod, y_train, x_test_prod, y_test)
print()
