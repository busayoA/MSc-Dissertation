import DataProcessor as dp
from TreeRNN import TreeRNN
import tensorflow as tf
from os.path import dirname, join

current_dir = dirname(__file__)

x_train, y_train, x_test, y_test = dp.getData(False)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


model = TreeRNN(x_train, y_train, [0], "", 0.0, 0)

x_train_usum, x_test_usum = [], []
x_train_umean, x_test_umean = [], []
x_train_umax, x_test_umax = [], []
x_train_umin, x_test_umin = [], []
x_train_uprod, x_test_uprod = [], []

for i in x_train:
    i = [i, i, i, i, i, i, i, i, i, i]
    i = tf.convert_to_tensor(i)

    uSum = model.segmentationLayer("sorted_sum", i)
    uSum = tf.reshape(uSum, (len(uSum[0]), 10))
    x_train_usum.append(uSum[0])

    uMean = model.segmentationLayer("sorted_mean", i)
    uMean = tf.reshape(uMean, (len(uMean[0]), 10))
    x_train_umean.append(uMean[0])

    uMax = model.segmentationLayer("sorted_max", i)
    uMax = tf.reshape(uMax, (len(uMax[0]), 10))
    x_train_umax.append(uMax[0])

    uMin = model.segmentationLayer("sorted_min", i)
    uMin = tf.reshape(uMin, (len(uMin[0]), 10))
    x_train_umin.append(uMin[0])

    uProd = model.segmentationLayer("sorted_prod", i)
    uProd = tf.reshape(uProd, (len(uProd[0]), 10))
    x_train_uprod.append(uProd[0])

for i in x_test:
    i = [i, i, i, i, i, i, i, i, i, i]
    i = tf.convert_to_tensor(i)


    uSum = model.segmentationLayer("sorted_sum", i)
    uSum = tf.reshape(uSum, (len(uSum[0]), 10))
    x_test_usum.append(uSum[0])

    uMean = model.segmentationLayer("sorted_mean", i)
    uMean = tf.reshape(uMean, (len(uMean[0]), 10))
    x_test_umean.append(uMean[0])

    uMax = model.segmentationLayer("sorted_max", i)
    uMax = tf.reshape(uMax, (len(uMax[0]), 10))
    x_test_umax.append(uMax[0])

    uMin = model.segmentationLayer("sorted_min", i)
    uMin = tf.reshape(uMin, (len(uMin[0]), 10))
    x_test_umin.append(uMin[0])

    uProd = model.segmentationLayer("sorted_prod", i)
    uProd = tf.reshape(uProd, (len(uProd[0]), 10))
    x_test_uprod.append(uProd[0])


x_train_usum = tf.convert_to_tensor(x_train_usum)
x_train_umean = tf.convert_to_tensor(x_train_umean)
x_train_umax = tf.convert_to_tensor(x_train_umax)
x_train_umin = tf.convert_to_tensor(x_train_umin)
x_train_uprod = tf.convert_to_tensor(x_train_uprod)

x_test_usum = tf.convert_to_tensor(x_test_usum)
x_test_umean = tf.convert_to_tensor(x_test_umean)
x_test_umax = tf.convert_to_tensor(x_test_umax)
x_test_umin = tf.convert_to_tensor(x_test_umin)
x_test_uprod = tf.convert_to_tensor(x_test_uprod)

filenameUSum = join(current_dir, "./Models/sorted_sum.hdf5")
filenameUMean = join(current_dir, "./Models/sorted_mean.hdf5")
filenameUMax = join(current_dir, "./Models/sorted_max.hdf5")
filenameUMin = join(current_dir, "./Models/sorted_min.hdf5")
filenameUProd = join(current_dir, "./Models/sorted_prod.hdf5")

""" RUN THE UNHASHED MODEL """

# USING RELU ACTIVATION
print("RUNNING RNN MODELS USING SORTED SEGMENTATION AND UNHASHED NODES")

# x_test_seg = tf.convert_to_tensor(x_test_seg)
print("UNSORTED SEGMENT SUM")
model.runRNNModel(x_train_usum, y_train, x_test_usum, y_test, filenameUSum)

print("UNSORTED SEGMENT MEAN")
model.runRNNModel(x_train_umean, y_train, x_test_umean, y_test, filenameUMean)

print("UNSORTED SEGMENT MAX")
model.runRNNModel(x_train_umax, y_train, x_test_umax, y_test, filenameUMax)

print("UNSORTED SEGMENT MIN")
model.runRNNModel(x_train_umin, y_train, x_test_umin, y_test, filenameUMin)

print("UNSORTED SEGMENT PROD")
model.runRNNModel(x_train_uprod, y_train, x_test_uprod, y_test, filenameUProd)
print()

