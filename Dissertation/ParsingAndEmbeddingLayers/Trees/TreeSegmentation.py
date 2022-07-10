import tensorflow as tf
import TreeDataProcessor as tdp
from TreeSegmentationLayer import TreeSegmentationLayer

segmentCount = 40
segmentationLayer = TreeSegmentationLayer()

def getUnsortedSegmentTrainData(hashed):
    x_train, y_train, x_test, y_test = tdp.getData(hashed) 
    y_train = tf.keras.utils.to_categorical(y_train)
    
    x_train_usum, x_train_umean, x_train_umax, x_train_umin, x_train_uprod = [], [], [], [], []

    for i in x_train:
        i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i,  i, i, i, i, i, i, i, i, i, i]
        i = tf.convert_to_tensor(i)

        uSum = segmentationLayer.segmentationLayer("unsorted_sum", i, segmentCount)
        uSum = tf.reshape(uSum, (len(uSum[0]), segmentCount))
        x_train_usum.append(uSum[0])

        uMean = segmentationLayer.segmentationLayer("unsorted_mean", i, segmentCount)
        uMean = tf.reshape(uMean, (len(uMean[0]), segmentCount))
        x_train_umean.append(uMean[0])

        uMax = segmentationLayer.segmentationLayer("unsorted_max", i, segmentCount)
        uMax = tf.reshape(uMax, (len(uMax[0]), segmentCount))
        x_train_umax.append(uMax[0])

        uMin = segmentationLayer.segmentationLayer("unsorted_min", i, segmentCount)
        uMin = tf.reshape(uMin, (len(uMin[0]), segmentCount))
        x_train_umin.append(uMin[0])

        uProd = segmentationLayer.segmentationLayer("unsorted_prod", i, segmentCount)
        uProd = tf.reshape(uProd, (len(uProd[0]), segmentCount))
        x_train_uprod.append(uProd[0])

    x_train_usum = tf.convert_to_tensor(x_train_usum)
    x_train_umean = tf.convert_to_tensor(x_train_umean)
    x_train_umax = tf.convert_to_tensor(x_train_umax)
    x_train_umin = tf.convert_to_tensor(x_train_umin)
    x_train_uprod = tf.convert_to_tensor(x_train_uprod)

    return x_train_usum, x_train_umean, x_train_umax, x_train_umin, x_train_uprod, y_train

def getUnsortedSegmentTestData(hashed):
    x_train, y_train, x_test, y_test = tdp.getData(hashed) # used for when we want to test using hashed data
    y_test = tf.keras.utils.to_categorical(y_test)

    x_test_usum, x_test_umean, x_test_umax, x_test_umin, x_test_uprod = [], [], [], [], []

    for i in x_test:
        i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i,  i, i, i, i, i, i, i, i, i, i]
        i = tf.convert_to_tensor(i)

        uSum = segmentationLayer.segmentationLayer("unsorted_sum", i, segmentCount)
        uSum = tf.reshape(uSum, (len(uSum[0]), segmentCount))
        x_test_usum.append(uSum[0])

        uMean = segmentationLayer.segmentationLayer("unsorted_mean", i, segmentCount)
        uMean = tf.reshape(uMean, (len(uMean[0]), segmentCount))
        x_test_umean.append(uMean[0])

        uMax = segmentationLayer.segmentationLayer("unsorted_max", i, segmentCount)
        uMax = tf.reshape(uMax, (len(uMax[0]), segmentCount))
        x_test_umax.append(uMax[0])

        uMin = segmentationLayer.segmentationLayer("unsorted_min", i, segmentCount)
        uMin = tf.reshape(uMin, (len(uMin[0]), segmentCount))
        x_test_umin.append(uMin[0])

        uProd = segmentationLayer.segmentationLayer("unsorted_prod", i, segmentCount)
        uProd = tf.reshape(uProd, (len(uProd[0]), segmentCount))
        x_test_uprod.append(uProd[0])

    x_test_usum = tf.convert_to_tensor(x_test_usum)
    x_test_umean = tf.convert_to_tensor(x_test_umean)
    x_test_umax = tf.convert_to_tensor(x_test_umax)
    x_test_umin = tf.convert_to_tensor(x_test_umin)
    x_test_uprod = tf.convert_to_tensor(x_test_uprod)

    return x_test_usum, x_test_umean, x_test_umax, x_test_umin, x_test_uprod, y_test


def getSortedSegmentTrainData(hashed):
    x_train, y_train, x_test, y_test = tdp.getData(hashed)
    y_train = tf.keras.utils.to_categorical(y_train)

    x_train_sum, x_train_mean, x_train_max, x_train_min, x_train_prod = [], [], [], [], []

    for i in x_train:
        i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i,  i, i, i, i, i, i, i, i, i, i]
        i = tf.convert_to_tensor(i)

        uSum = segmentationLayer.segmentationLayer("sorted_sum", i, segmentCount)
        uSum = tf.reshape(uSum, (len(uSum[0]), segmentCount))
        x_train_sum.append(uSum[0])

        uMean = segmentationLayer.segmentationLayer("sorted_mean", i, segmentCount)
        uMean = tf.reshape(uMean, (len(uMean[0]), segmentCount))
        x_train_mean.append(uMean[0])

        uMax = segmentationLayer.segmentationLayer("sorted_max", i, segmentCount)
        uMax = tf.reshape(uMax, (len(uMax[0]), segmentCount))
        x_train_max.append(uMax[0])

        uMin = segmentationLayer.segmentationLayer("sorted_min", i, segmentCount)
        uMin = tf.reshape(uMin, (len(uMin[0]), segmentCount))
        x_train_min.append(uMin[0])

        uProd = segmentationLayer.segmentationLayer("sorted_prod", i, segmentCount)
        uProd = tf.reshape(uProd, (len(uProd[0]), segmentCount))
        x_train_prod.append(uProd[0])
    
    x_train_sum = tf.convert_to_tensor(x_train_sum)
    x_train_mean = tf.convert_to_tensor(x_train_mean)
    x_train_max = tf.convert_to_tensor(x_train_max)
    x_train_min = tf.convert_to_tensor(x_train_min)
    x_train_prod = tf.convert_to_tensor(x_train_prod)

    return x_train_sum, x_train_mean, x_train_max, x_train_min, x_train_prod, y_train


def getSortedSegmentTestData(hashed):
    x_train, y_train, x_test, y_test = tdp.getData(hashed)
    y_test = tf.keras.utils.to_categorical(y_test)

    x_test_sum, x_test_mean, x_test_max, x_test_min, x_test_prod = [], [], [], [], []
    for i in x_test:
        i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i,  i, i, i, i, i, i, i, i, i, i]
        i = tf.convert_to_tensor(i)

        uSum = segmentationLayer.segmentationLayer("sorted_sum", i, segmentCount)
        uSum = tf.reshape(uSum, (len(uSum[0]), segmentCount))
        x_test_sum.append(uSum[0])

        uMean = segmentationLayer.segmentationLayer("sorted_mean", i, segmentCount)
        uMean = tf.reshape(uMean, (len(uMean[0]), segmentCount))
        x_test_mean.append(uMean[0])

        uMax = segmentationLayer.segmentationLayer("sorted_max", i, segmentCount)
        uMax = tf.reshape(uMax, (len(uMax[0]), segmentCount))
        x_test_max.append(uMax[0])

        uMin = segmentationLayer.segmentationLayer("sorted_min", i, segmentCount)
        uMin = tf.reshape(uMin, (len(uMin[0]), segmentCount))
        x_test_min.append(uMin[0])

        uProd = segmentationLayer.segmentationLayer("sorted_prod", i, segmentCount)
        uProd = tf.reshape(uProd, (len(uProd[0]), segmentCount))
        x_test_prod.append(uProd[0])


    x_test_sum = tf.convert_to_tensor(x_test_sum)
    x_test_mean = tf.convert_to_tensor(x_test_mean)
    x_test_max = tf.convert_to_tensor(x_test_max)
    x_test_min = tf.convert_to_tensor(x_test_min)
    x_test_prod = tf.convert_to_tensor(x_test_prod)

    return x_test_sum, x_test_mean, x_test_max, x_test_min, x_test_prod, y_test