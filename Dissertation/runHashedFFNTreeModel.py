import tensorflow as tf
import TreeDataProcessor as tdp
from MLP import MLP
from SegmentationLayer import SegmentationLayer
import TreeSegmentation as seg

x_train_usum =  seg.x_train_usum
x_train_umean = seg.x_train_umean
x_train_umax = seg.x_train_umax
x_train_umin = seg.x_train_umin
x_train_uprod = seg.x_train_uprod

x_test_usum = seg.x_test_usum
x_test_umean = seg.x_test_umean
x_test_umax = seg.x_test_umax
x_test_umin = seg.x_test_umin
x_test_uprod = seg.x_test_uprod

x_train_sum = seg.x_train_sum
x_train_mean = seg.x_train_mean
x_train_max = seg.x_train_max
x_train_min = seg.x_train_min
x_train_prod = seg.x_train_prod

x_test_sum = seg.x_test_sum
x_test_mean = seg.x_test_mean
x_test_max = seg.x_test_max
x_test_min = seg.x_test_min
x_test_prod = seg.x_test_prod

y_train = seg.y_train
y_test = seg.y_test

segmentCount = 40
segmentationLayer = SegmentationLayer()
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
model1c = MLP(x_train_usum, y_train, layers, "logsigmoid", lr, epochs)
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
model2c = MLP(x_train_usum, y_train, layers, "logsigmoid", lr, epochs)
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
