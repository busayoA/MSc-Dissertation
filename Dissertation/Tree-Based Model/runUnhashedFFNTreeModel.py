import TreeDataProcessor as dp
from TreeNN import TreeNN
import tensorflow as tf
from os.path import dirname, join

current_dir = dirname(__file__)

x_train, y_train, x_test, y_test = dp.getData(False)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

segmentCount = 40

model = TreeNN(x_train, y_train, [0], "", 0.0, 0)

x_train_usum, x_test_usum = [], []
x_train_umean, x_test_umean = [], []
x_train_umax, x_test_umax = [], []
x_train_umin, x_test_umin = [], []
x_train_uprod, x_test_uprod = [], []

for i in x_train:
    # i = [i, i, i, i, i, i, i, i, i, i]
    # i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i]
    i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i,  i, i, i, i, i, i, i, i, i, i]
    i = tf.convert_to_tensor(i)

    uSum = model.segmentationLayer("unsorted_sum", i, segmentCount)
    uSum = tf.reshape(uSum, (len(uSum[0]), segmentCount))
    x_train_usum.append(uSum[0])

    uMean = model.segmentationLayer("unsorted_mean", i, segmentCount)
    uMean = tf.reshape(uMean, (len(uMean[0]), segmentCount))
    x_train_umean.append(uMean[0])

    uMax = model.segmentationLayer("unsorted_max", i, segmentCount)
    uMax = tf.reshape(uMax, (len(uMax[0]), segmentCount))
    x_train_umax.append(uMax[0])

    uMin = model.segmentationLayer("unsorted_min", i, segmentCount)
    uMin = tf.reshape(uMin, (len(uMin[0]), segmentCount))
    x_train_umin.append(uMin[0])

    uProd = model.segmentationLayer("unsorted_prod", i, segmentCount)
    uProd = tf.reshape(uProd, (len(uProd[0]), segmentCount))
    x_train_uprod.append(uProd[0])

for i in x_test:
        # i = [i, i, i, i, i, i, i, i, i, i]
    # i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i]
    i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i,  i, i, i, i, i, i, i, i, i, i]
    i = tf.convert_to_tensor(i)

    uSum = model.segmentationLayer("unsorted_sum", i, segmentCount)
    uSum = tf.reshape(uSum, (len(uSum[0]), segmentCount))
    x_test_usum.append(uSum[0])

    uMean = model.segmentationLayer("unsorted_mean", i, segmentCount)
    uMean = tf.reshape(uMean, (len(uMean[0]), segmentCount))
    x_test_umean.append(uMean[0])

    uMax = model.segmentationLayer("unsorted_max", i, segmentCount)
    uMax = tf.reshape(uMax, (len(uMax[0]), segmentCount))
    x_test_umax.append(uMax[0])

    uMin = model.segmentationLayer("unsorted_min", i, segmentCount)
    uMin = tf.reshape(uMin, (len(uMin[0]), segmentCount))
    x_test_umin.append(uMin[0])

    uProd = model.segmentationLayer("unsorted_prod", i, segmentCount)
    uProd = tf.reshape(uProd, (len(uProd[0]), segmentCount))
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

# USING RELU ACTIVATION
print("RUNNING RNN MODELS USING UNSORTED SEGMENTATION AND UNHASHED NODES")

# x_test_seg = tf.convert_to_tensor(x_test_seg)
print("UNSORTED SEGMENT SUM AND RELU")
model1a = TreeNN(x_train, y_train, [segmentCount, 64, 64, 2], "relu", 0.05, 30)
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
model1b = TreeNN(x_train, y_train, [segmentCount, 64, 64, 2], "tanh", 0.05, 30)
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
model1c = TreeNN(x_train, y_train, [segmentCount, 64, 64, 2], "logsigmoid", 0.05, 30)
print("UNSORTED SEGMENT MEAN AND LOGSIGMOID")
model1c.runFFModel(x_train_umean, y_train, x_test_umean, y_test)
print("UNSORTED SEGMENT MAX AND LOGSIGMOID")
model1c.runFFModel(x_train_umax, y_train, x_test_umax, y_test)
print("UNSORTED SEGMENT MIN AND LOGSIGMOID")
model1c.runFFModel(x_train_umin, y_train, x_test_umin, y_test)
print("UNSORTED SEGMENT PROD AND LOGSIGMOID")
model1c.runFFModel(x_train_uprod, y_train, x_test_uprod, y_test)
print()




x_train_usum, x_test_usum = [], []
x_train_umean, x_test_umean = [], []
x_train_umax, x_test_umax = [], []
x_train_umin, x_test_umin = [], []
x_train_uprod, x_test_uprod = [], []

for i in x_train:
    # i = [i, i, i, i, i, i, i, i, i, i]
    # i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i]
    i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i,  i, i, i, i, i, i, i, i, i, i]
    i = tf.convert_to_tensor(i)

    uSum = model.segmentationLayer("sorted_sum", i, segmentCount)
    uSum = tf.reshape(uSum, (len(uSum[0]), segmentCount))
    x_train_usum.append(uSum[0])

    uMean = model.segmentationLayer("sorted_mean", i, segmentCount)
    uMean = tf.reshape(uMean, (len(uMean[0]), segmentCount))
    x_train_umean.append(uMean[0])

    uMax = model.segmentationLayer("sorted_max", i, segmentCount)
    uMax = tf.reshape(uMax, (len(uMax[0]), segmentCount))
    x_train_umax.append(uMax[0])

    uMin = model.segmentationLayer("sorted_min", i, segmentCount)
    uMin = tf.reshape(uMin, (len(uMin[0]), segmentCount))
    x_train_umin.append(uMin[0])

    uProd = model.segmentationLayer("sorted_prod", i, segmentCount)
    uProd = tf.reshape(uProd, (len(uProd[0]), segmentCount))
    x_train_uprod.append(uProd[0])

for i in x_test:
    # i = [i, i, i, i, i, i, i, i, i, i]
    # i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i]
    i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i,  i, i, i, i, i, i, i, i, i, i]
    i = tf.convert_to_tensor(i)


    uSum = model.segmentationLayer("sorted_sum", i, segmentCount)
    uSum = tf.reshape(uSum, (len(uSum[0]), segmentCount))
    x_test_usum.append(uSum[0])

    uMean = model.segmentationLayer("sorted_mean", i, segmentCount)
    uMean = tf.reshape(uMean, (len(uMean[0]), segmentCount))
    x_test_umean.append(uMean[0])

    uMax = model.segmentationLayer("sorted_max", i, segmentCount)
    uMax = tf.reshape(uMax, (len(uMax[0]), segmentCount))
    x_test_umax.append(uMax[0])

    uMin = model.segmentationLayer("sorted_min", i, segmentCount)
    uMin = tf.reshape(uMin, (len(uMin[0]), segmentCount))
    x_test_umin.append(uMin[0])

    uProd = model.segmentationLayer("sorted_prod", i, segmentCount)
    uProd = tf.reshape(uProd, (len(uProd[0]), segmentCount))
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


print("RUNNING RNN MODELS USING SORTED SEGMENTATION AND UNHASHED NODES")
# USING RELU ACTIVATION
print("SORTED SEGMENT SUM AND RELU")
model2a = TreeNN(x_train, y_train, [segmentCount, 64, 64, 2], "relu", 0.05, 30)
model2a.runFFModel(x_train_usum, y_train, x_test_usum, y_test)
print("SORTED SEGMENT MEAN AND RELU")
model2a.runFFModel(x_train_umean, y_train, x_test_umean, y_test)
print("SORTED SEGMENT MAX AND RELU")
model2a.runFFModel(x_train_umax, y_train, x_test_umax, y_test)
print("SORTED SEGMENT MIN AND RELU")
model2a.runFFModel(x_train_umin, y_train, x_test_umin, y_test)
print("SORTED SEGMENT PROD AND RELU")
model2a.runFFModel(x_train_uprod, y_train, x_test_uprod, y_test)
print()


# USING TANH ACTIVATION
print("SORTED SEGMENT SUM AND TANH")
model2b = TreeNN(x_train, y_train, [segmentCount, 64, 64, 2], "tanh", 0.05, 30)
model2b.runFFModel(x_train_usum, y_train, x_test_usum, y_test)
print("SORTED SEGMENT MEAN AND TANH")
model2b.runFFModel(x_train_umean, y_train, x_test_umean, y_test)
print("SORTED SEGMENT MAX AND TANH")
model2b.runFFModel(x_train_umax, y_train, x_test_umax, y_test)
print("SORTED SEGMENT MIN AND TANH")
model2b.runFFModel(x_train_umin, y_train, x_test_umin, y_test)
print("SORTED SEGMENT PROD AND TANH")
model2b.runFFModel(x_train_uprod, y_train, x_test_uprod, y_test)
print()


# USING LOGSIGMOID ACTIVATION
print("SORTED SEGMENT SUM AND LOGSIGMOID")
model2c = TreeNN(x_train, y_train, [segmentCount, 64, 64, 2], "logsigmoid", 0.05, 30)
model2c.runFFModel(x_train_usum, y_train, x_test_usum, y_test)
print("SORTED SEGMENT MEAN AND LOGSIGMOID")
model2c.runFFModel(x_train_umean, y_train, x_test_umean, y_test)
print("SORTED SEGMENT MAX AND LOGSIGMOID")
model2c.runFFModel(x_train_umax, y_train, x_test_umax, y_test)
print("SORTED SEGMENT MIN AND LOGSIGMOID")
model2c.runFFModel(x_train_umin, y_train, x_test_umin, y_test)
print("SORTED SEGMENT PROD AND LOGSIGMOID")
model2c.runFFModel(x_train_uprod, y_train, x_test_uprod, y_test)
print()
