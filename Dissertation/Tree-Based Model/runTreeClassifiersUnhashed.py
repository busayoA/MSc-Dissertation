import TreeClassifiers 
import TreeDataProcessor as dp
from TreeNN import TreeNN
import tensorflow as tf
from sklearn.metrics import accuracy_score
from os.path import dirname, join

current_dir = dirname(__file__)

x_train, y_train, x_test, y_test = dp.getData(False)

segmentCount = 40
model = TreeNN(x_train, y_train, [0], "", 0.0, 0)

x_train_usum, x_test_usum = [], []
x_train_umean, x_test_umean = [], []
x_train_umax, x_test_umax = [], []
x_train_umin, x_test_umin = [], []
x_train_uprod, x_test_uprod = [], []

"""USING UNHASHED NODES AND UNSORTED SEGMENTATION"""
for i in x_train:
    # i = [i, i, i, i, i, i, i, i, i, i]
    # i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i]
    i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i,  i, i, i, i, i, i, i, i, i, i]
    i = tf.convert_to_tensor(i)

    uSum = model.segmentationLayer("unsorted_sum", i, segmentCount)
    uSum = tf.reshape(uSum, (len(uSum[0]), segmentCount))
    x_train_usum.append(list(uSum[0].numpy()))

    uMean = model.segmentationLayer("unsorted_mean", i, segmentCount)
    uMean = tf.reshape(uMean, (len(uMean[0]), segmentCount))
    x_train_umean.append(list(uMean[0].numpy()))

    uMax = model.segmentationLayer("unsorted_max", i, segmentCount)
    uMax = tf.reshape(uMax, (len(uMax[0]), segmentCount))
    x_train_umax.append(list(uMax[0].numpy()))

    uMin = model.segmentationLayer("unsorted_min", i, segmentCount)
    uMin = tf.reshape(uMin, (len(uMin[0]), segmentCount))
    x_train_umin.append(list(uMin[0].numpy()))

    uProd = model.segmentationLayer("unsorted_prod", i, segmentCount)
    uProd = tf.reshape(uProd, (len(uProd[0]), segmentCount))
    x_train_uprod.append(list(uProd[0].numpy()))

for i in x_test:
     # i = [i, i, i, i, i, i, i, i, i, i]
    # i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i]
    i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i,  i, i, i, i, i, i, i, i, i, i]
    i = tf.convert_to_tensor(i)

    uSum = model.segmentationLayer("unsorted_sum", i, segmentCount)
    uSum = tf.reshape(uSum, (len(uSum[0]), segmentCount))
    x_test_usum.append(list(uSum[0].numpy()))

    uMean = model.segmentationLayer("unsorted_mean", i, segmentCount)
    uMean = tf.reshape(uMean, (len(uMean[0]), segmentCount))
    x_test_umean.append(list(uMean[0].numpy()))

    uMax = model.segmentationLayer("unsorted_max", i, segmentCount)
    uMax = tf.reshape(uMax, (len(uMax[0]), segmentCount))
    x_test_umax.append(list(uMax[0].numpy()))

    uMin = model.segmentationLayer("unsorted_min", i, segmentCount)
    uMin = tf.reshape(uMin, (len(uMin[0]), segmentCount))
    x_test_umin.append(list(uMin[0].numpy()))

    uProd = model.segmentationLayer("unsorted_prod", i, segmentCount)
    uProd = tf.reshape(uProd, (len(uProd[0]), segmentCount))
    x_test_uprod.append(list(uProd[0].numpy()))

def convertYTo1d(y):
    y_values = []
    for i in y:
        newValue = int(i[0])
        y_values.append(newValue)

    return y_values

y_train = convertYTo1d(y_train)
y_test = convertYTo1d(y_test)

sgdUSumUnhashed = TreeClassifiers.SGDClassify(x_train_usum, y_train, x_test_usum)
svmUSumUnhashed = TreeClassifiers.SVMClassifier(x_train_usum, y_train, x_test_usum)
rfUSumUnhashed = TreeClassifiers.rfClassify(x_train_usum, y_train, x_test_usum)
# nbUsum = TreeClassifiers.nbClassify(x_train_usum, y_train, x_test_usum)
sgdAccuracyUSumUnhashed = accuracy_score(sgdUSumUnhashed, y_test)
svmAccuracyUSumUnhashed = accuracy_score(svmUSumUnhashed, y_test)
rfAccuracyUSumUnhashed = accuracy_score(rfUSumUnhashed, y_test)


sgdUMeanUnhashed = TreeClassifiers.SGDClassify(x_train_umean, y_train, x_test_umean)
svmUMeanUnhashed = TreeClassifiers.SVMClassifier(x_train_umean, y_train, x_test_umean)
rfUMeanUnhashed = TreeClassifiers.rfClassify(x_train_umean, y_train, x_test_umean)
sgdAccuracyUMeanUnhashed = accuracy_score(sgdUMeanUnhashed, y_test)
svmAccuracyUMeanUnhashed = accuracy_score(svmUMeanUnhashed, y_test)
rfAccuracyUMeanUnhashed = accuracy_score(rfUMeanUnhashed, y_test)

sgdUMaxUnhashed = TreeClassifiers.SGDClassify(x_train_umax, y_train, x_test_umax)
svmUMaxUnhashed = TreeClassifiers.SVMClassifier(x_train_umax, y_train, x_test_umax)
rfUMaxUnhashed = TreeClassifiers.rfClassify(x_train_umax, y_train, x_test_umax)
sgdAccuracyUMaxUnhashed = accuracy_score(sgdUMaxUnhashed, y_test)
svmAccuracyUMaxUnhashed = accuracy_score(svmUMaxUnhashed, y_test)
rfAccuracyUMaxUnhashed = accuracy_score(rfUMaxUnhashed, y_test)

sgdUMinUnhashed = TreeClassifiers.SGDClassify(x_train_umin, y_train, x_test_umin)
svmUMinUnhashed = TreeClassifiers.SVMClassifier(x_train_umin, y_train, x_test_umin)
rfUMinUnhashed = TreeClassifiers.rfClassify(x_train_umin, y_train, x_test_umin)
sgdAccuracyUMinUnhashed = accuracy_score(sgdUMinUnhashed, y_test)
svmAccuracyUMinUnhashed = accuracy_score(svmUMinUnhashed, y_test)
rfAccuracyUMinUnhashed = accuracy_score(rfUMinUnhashed, y_test)

sgdUProdUnhashed = TreeClassifiers.SGDClassify(x_train_uprod, y_train, x_test_uprod)
svmUProdUnhashed = TreeClassifiers.SVMClassifier(x_train_uprod, y_train, x_test_uprod)
rfUProdUnhashed = TreeClassifiers.rfClassify(x_train_uprod, y_train, x_test_uprod)
sgdAccuracyUProdUnhashed = accuracy_score(sgdUProdUnhashed, y_test)
svmAccuracyUProdUnhashed = accuracy_score(svmUProdUnhashed, y_test)
rfAccuracyUProdUnhashed = accuracy_score(rfUProdUnhashed, y_test)





"""USING UNHASHED NODES AND SORTED SEGMENTATION"""
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
    x_train_usum.append(list(uSum[0].numpy()))

    uMean = model.segmentationLayer("sorted_mean", i, segmentCount)
    uMean = tf.reshape(uMean, (len(uMean[0]), segmentCount))
    x_train_umean.append(list(uMean[0].numpy()))

    uMax = model.segmentationLayer("sorted_max", i, segmentCount)
    uMax = tf.reshape(uMax, (len(uMax[0]), segmentCount))
    x_train_umax.append(list(uMax[0].numpy()))

    uMin = model.segmentationLayer("sorted_min", i, segmentCount)
    uMin = tf.reshape(uMin, (len(uMin[0]), segmentCount))
    x_train_umin.append(list(uMin[0].numpy()))

    uProd = model.segmentationLayer("sorted_prod", i, segmentCount)
    uProd = tf.reshape(uProd, (len(uProd[0]), segmentCount))
    x_train_uprod.append(list(uProd[0].numpy()))

for i in x_test:
     # i = [i, i, i, i, i, i, i, i, i, i]
    # i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i]
    i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i,  i, i, i, i, i, i, i, i, i, i]
    i = tf.convert_to_tensor(i)

    uSum = model.segmentationLayer("sorted_sum", i, segmentCount)
    uSum = tf.reshape(uSum, (len(uSum[0]), segmentCount))
    x_test_usum.append(list(uSum[0].numpy()))

    uMean = model.segmentationLayer("sorted_mean", i, segmentCount)
    uMean = tf.reshape(uMean, (len(uMean[0]), segmentCount))
    x_test_umean.append(list(uMean[0].numpy()))

    uMax = model.segmentationLayer("sorted_max", i, segmentCount)
    uMax = tf.reshape(uMax, (len(uMax[0]), segmentCount))
    x_test_umax.append(list(uMax[0].numpy()))

    uMin = model.segmentationLayer("sorted_min", i, segmentCount)
    uMin = tf.reshape(uMin, (len(uMin[0]), segmentCount))
    x_test_umin.append(list(uMin[0].numpy()))

    uProd = model.segmentationLayer("sorted_prod", i, segmentCount)
    uProd = tf.reshape(uProd, (len(uProd[0]), segmentCount))
    x_test_uprod.append(list(uProd[0].numpy()))

sgdSumUnhashed = TreeClassifiers.SGDClassify(x_train_usum, y_train, x_test_usum)
svmSumUnhashed = TreeClassifiers.SVMClassifier(x_train_usum, y_train, x_test_usum)
rfSumUnhashed = TreeClassifiers.rfClassify(x_train_usum, y_train, x_test_usum)
# nbUsum = TreeClassifiers.nbClassify(x_train_usum, y_train, x_test_usum)
sgdAccuracySumUnhashed = accuracy_score(sgdSumUnhashed, y_test)
svmAccuracySumUnhashed = accuracy_score(svmSumUnhashed, y_test)
rfAccuracySumUnhashed = accuracy_score(rfSumUnhashed, y_test)


sgdMeanUnhashed = TreeClassifiers.SGDClassify(x_train_umean, y_train, x_test_umean)
svmMeanUnhashed = TreeClassifiers.SVMClassifier(x_train_umean, y_train, x_test_umean)
rfMeanUnhashed = TreeClassifiers.rfClassify(x_train_umean, y_train, x_test_umean)
sgdAccuracyMeanUnhashed = accuracy_score(sgdMeanUnhashed, y_test)
svmAccuracyMeanUnhashed = accuracy_score(svmMeanUnhashed, y_test)
rfAccuracyMeanUnhashed = accuracy_score(rfMeanUnhashed, y_test)

sgdMaxUnhashed = TreeClassifiers.SGDClassify(x_train_umax, y_train, x_test_umax)
svmMaxUnhashed = TreeClassifiers.SVMClassifier(x_train_umax, y_train, x_test_umax)
rfMaxUnhashed = TreeClassifiers.rfClassify(x_train_umax, y_train, x_test_umax)
sgdAccuracyMaxUnhashed = accuracy_score(sgdMaxUnhashed, y_test)
svmAccuracyMaxUnhashed = accuracy_score(svmMaxUnhashed, y_test)
rfAccuracyMaxUnhashed = accuracy_score(rfMaxUnhashed, y_test)

sgdMinUnhashed = TreeClassifiers.SGDClassify(x_train_umin, y_train, x_test_umin)
svmMinUnhashed = TreeClassifiers.SVMClassifier(x_train_umin, y_train, x_test_umin)
rfMinUnhashed = TreeClassifiers.rfClassify(x_train_umin, y_train, x_test_umin)
sgdAccuracyMinUnhashed = accuracy_score(sgdMinUnhashed, y_test)
svmAccuracyMinUnhashed = accuracy_score(svmMinUnhashed, y_test)
rfAccuracyMinUnhashed = accuracy_score(rfMinUnhashed, y_test)

sgdProdUnhashed = TreeClassifiers.SGDClassify(x_train_uprod, y_train, x_test_uprod)
svmProdUnhashed = TreeClassifiers.SVMClassifier(x_train_uprod, y_train, x_test_uprod)
rfProdUnhashed = TreeClassifiers.rfClassify(x_train_uprod, y_train, x_test_uprod)
sgdAccuracyProdUnhashed = accuracy_score(sgdProdUnhashed, y_test)
svmAccuracyProdUnhashed = accuracy_score(svmProdUnhashed, y_test)
rfAccuracyProdUnhashed = accuracy_score(rfProdUnhashed, y_test)

print("Unsorted Sum Segmentation: \nSGD:", sgdAccuracyUSumUnhashed, "SVM:", svmAccuracyUSumUnhashed, 
"Random Forest", rfAccuracyUSumUnhashed)
print("Sorted Sum Segmentation: \nSGD:", sgdAccuracySumUnhashed, "SVM:", svmAccuracySumUnhashed, 
"Random Forest", rfAccuracySumUnhashed)
print()

print("Unsorted Mean Segmentation: \nSGD:", sgdAccuracyUMeanUnhashed, "SVM:", svmAccuracyUMeanUnhashed, 
"Random Forest", rfAccuracyUMeanUnhashed)
print("Sorted Mean Segmentation: \nSGD:", sgdAccuracyMeanUnhashed, "SVM:", svmAccuracyMeanUnhashed, 
"Random Forest", rfAccuracyMeanUnhashed)
print()

print("Unsorted Max Segmentation: \nSGD:", sgdAccuracyUMaxUnhashed, "SVM:", svmAccuracyUMaxUnhashed, 
"Random Forest", rfAccuracyUMaxUnhashed)
print("Sorted Max Segmentation: \nSGD:", sgdAccuracyMaxUnhashed, "SVM:", svmAccuracyMaxUnhashed, 
"Random Forest", rfAccuracyMaxUnhashed)
print()

print("Unsorted Min Segmentation: \nSGD:", sgdAccuracyUMinUnhashed, "SVM:", svmAccuracyUMinUnhashed, 
"Random Forest", rfAccuracyUMinUnhashed)
print("Sorted Min Segmentation: \nSGD:", sgdAccuracyMinUnhashed, "SVM:", svmAccuracyMinUnhashed, 
"Random Forest", rfAccuracyMinUnhashed)
print()

print("Unsorted Prod Segmentation: \nSGD:", sgdAccuracyUProdUnhashed, "SVM:", svmAccuracyUProdUnhashed, 
"Random Forest", rfAccuracyUProdUnhashed)
print("Sorted Prod Segmentation: \nSGD:", sgdAccuracyProdUnhashed, "SVM:", svmAccuracyProdUnhashed, 
"Random Forest", rfAccuracyProdUnhashed)
print()







