import TreeClassifiers 
import TreeDataProcessor as dp
from TreeNN import TreeNN
import tensorflow as tf
from sklearn.metrics import accuracy_score

x_train, y_train, x_test, y_test = dp.getData(True)

def convertYTo1d(y):
    y_values = []
    for i in y:
        newValue = int(i[0])
        y_values.append(newValue)

    return y_values

y_train = convertYTo1d(y_train)
y_test = convertYTo1d(y_test)

segmentCount = 40
model = TreeNN(x_train, y_train, [0], "", 0.0, 0)

x_train_usum, x_test_usum = [], []
x_train_umean, x_test_umean = [], []
x_train_umax, x_test_umax = [], []
x_train_umin, x_test_umin = [], []
x_train_uprod, x_test_uprod = [], []

x_train_sum, x_test_sum = [], []
x_train_mean, x_test_mean = [], []
x_train_max, x_test_max = [], []
x_train_min, x_test_min = [], []
x_train_prod, x_test_prod = [], []

"""USING HASHED NODES AND UNSORTED SEGMENTATION"""
for i in x_train:
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


for i in x_train:
    # i = [i, i, i, i, i, i, i, i, i, i]
    # i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i]
    i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i,  i, i, i, i, i, i, i, i, i, i]
    i = tf.convert_to_tensor(i)

    uSum = model.segmentationLayer("sorted_sum", i, segmentCount)
    uSum = tf.reshape(uSum, (len(uSum[0]), segmentCount))
    x_train_sum.append(list(uSum[0].numpy()))

    uMean = model.segmentationLayer("sorted_mean", i, segmentCount)
    uMean = tf.reshape(uMean, (len(uMean[0]), segmentCount))
    x_train_mean.append(list(uMean[0].numpy()))

    uMax = model.segmentationLayer("sorted_max", i, segmentCount)
    uMax = tf.reshape(uMax, (len(uMax[0]), segmentCount))
    x_train_max.append(list(uMax[0].numpy()))

    uMin = model.segmentationLayer("sorted_min", i, segmentCount)
    uMin = tf.reshape(uMin, (len(uMin[0]), segmentCount))
    x_train_min.append(list(uMin[0].numpy()))

    uProd = model.segmentationLayer("sorted_prod", i, segmentCount)
    uProd = tf.reshape(uProd, (len(uProd[0]), segmentCount))
    x_train_prod.append(list(uProd[0].numpy()))

for i in x_test:
     # i = [i, i, i, i, i, i, i, i, i, i]
    # i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i]
    i = [i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i,  i, i, i, i, i, i, i, i, i, i]
    i = tf.convert_to_tensor(i)

    uSum = model.segmentationLayer("sorted_sum", i, segmentCount)
    uSum = tf.reshape(uSum, (len(uSum[0]), segmentCount))
    x_test_sum.append(list(uSum[0].numpy()))

    uMean = model.segmentationLayer("sorted_mean", i, segmentCount)
    uMean = tf.reshape(uMean, (len(uMean[0]), segmentCount))
    x_test_mean.append(list(uMean[0].numpy()))

    uMax = model.segmentationLayer("sorted_max", i, segmentCount)
    uMax = tf.reshape(uMax, (len(uMax[0]), segmentCount))
    x_test_max.append(list(uMax[0].numpy()))

    uMin = model.segmentationLayer("sorted_min", i, segmentCount)
    uMin = tf.reshape(uMin, (len(uMin[0]), segmentCount))
    x_test_min.append(list(uMin[0].numpy()))

    uProd = model.segmentationLayer("sorted_prod", i, segmentCount)
    uProd = tf.reshape(uProd, (len(uProd[0]), segmentCount))
    x_test_prod.append(list(uProd[0].numpy()))



sgdUSumHashed = TreeClassifiers.SGDClassify(x_train_usum, y_train, x_test_usum)
svmUSumHashed = TreeClassifiers.SVMClassifier(x_train_usum, y_train, x_test_usum)
rfUSumHashed = TreeClassifiers.rfClassify(x_train_usum, y_train, x_test_usum)
nbUSumHashed = TreeClassifiers.nbClassify(x_train_usum, y_train, x_test_usum)

sgdAccuracyUSumHashed = accuracy_score(sgdUSumHashed, y_test)
svmAccuracyUSumHashed = accuracy_score(svmUSumHashed, y_test)
rfAccuracyUSumHashed = accuracy_score(rfUSumHashed, y_test)
nbAccuracyUSumHashed = accuracy_score(nbUSumHashed, y_test)


sgdUMeanHashed = TreeClassifiers.SGDClassify(x_train_umean, y_train, x_test_umean)
svmUMeanHashed = TreeClassifiers.SVMClassifier(x_train_umean, y_train, x_test_umean)
rfUMeanHashed = TreeClassifiers.rfClassify(x_train_umean, y_train, x_test_umean)
nbUMeanHashed = TreeClassifiers.nbClassify(x_train_umean, y_train, x_test_umean)

sgdAccuracyUMeanHashed = accuracy_score(sgdUMeanHashed, y_test)
svmAccuracyUMeanHashed = accuracy_score(svmUMeanHashed, y_test)
rfAccuracyUMeanHashed = accuracy_score(rfUMeanHashed, y_test)
nbAccuracyUMeanHashed = accuracy_score(nbUMeanHashed, y_test)



sgdUMaxHashed = TreeClassifiers.SGDClassify(x_train_umax, y_train, x_test_umax)
svmUMaxHashed = TreeClassifiers.SVMClassifier(x_train_umax, y_train, x_test_umax)
rfUMaxHashed = TreeClassifiers.rfClassify(x_train_umax, y_train, x_test_umax)
nbUMaxHashed = TreeClassifiers.nbClassify(x_train_umax, y_train, x_test_umax)

sgdAccuracyUMaxHashed = accuracy_score(sgdUMaxHashed, y_test)
svmAccuracyUMaxHashed = accuracy_score(svmUMaxHashed, y_test)
rfAccuracyUMaxHashed = accuracy_score(rfUMaxHashed, y_test)
nbAccuracyUMaxHashed = accuracy_score(nbUMaxHashed, y_test)



sgdUMinHashed = TreeClassifiers.SGDClassify(x_train_umin, y_train, x_test_umin)
svmUMinHashed = TreeClassifiers.SVMClassifier(x_train_umin, y_train, x_test_umin)
rfUMinHashed = TreeClassifiers.rfClassify(x_train_umin, y_train, x_test_umin)
nbUMinHashed = TreeClassifiers.nbClassify(x_train_umin, y_train, x_test_umin)

sgdAccuracyUMinHashed = accuracy_score(sgdUMinHashed, y_test)
svmAccuracyUMinHashed = accuracy_score(svmUMinHashed, y_test)
rfAccuracyUMinHashed = accuracy_score(rfUMinHashed, y_test)
nbAccuracyUMinHashed = accuracy_score(nbUMinHashed, y_test)



sgdUProdHashed = TreeClassifiers.SGDClassify(x_train_uprod, y_train, x_test_uprod)
svmUProdHashed = TreeClassifiers.SVMClassifier(x_train_uprod, y_train, x_test_uprod)
rfUProdHashed = TreeClassifiers.rfClassify(x_train_uprod, y_train, x_test_uprod)
nbUProdHashed = TreeClassifiers.nbClassify(x_train_uprod, y_train, x_test_uprod)

sgdAccuracyUProdHashed = accuracy_score(sgdUProdHashed, y_test)
svmAccuracyUProdHashed = accuracy_score(svmUProdHashed, y_test)
rfAccuracyUProdHashed = accuracy_score(rfUProdHashed, y_test)
nbAccuracyUProdHashed = accuracy_score(nbUProdHashed, y_test)





"""USING HASHED NODES AND SORTED SEGMENTATION"""

sgdSumHashed = TreeClassifiers.SGDClassify(x_train_sum, y_train, x_test_sum)
svmSumHashed = TreeClassifiers.SVMClassifier(x_train_sum, y_train, x_test_sum)
rfSumHashed = TreeClassifiers.rfClassify(x_train_sum, y_train, x_test_sum)
nbSumHashed = TreeClassifiers.nbClassify(x_train_sum, y_train, x_test_sum)

sgdAccuracySumHashed = accuracy_score(sgdSumHashed, y_test)
svmAccuracySumHashed = accuracy_score(svmSumHashed, y_test)
rfAccuracySumHashed = accuracy_score(rfSumHashed, y_test)
nbAccuracySumHashed = accuracy_score(nbSumHashed, y_test)


sgdMeanHashed = TreeClassifiers.SGDClassify(x_train_mean, y_train, x_test_mean)
svmMeanHashed = TreeClassifiers.SVMClassifier(x_train_mean, y_train, x_test_mean)
rfMeanHashed = TreeClassifiers.rfClassify(x_train_mean, y_train, x_test_mean)
nbMeanHashed = TreeClassifiers.nbClassify(x_train_mean, y_train, x_test_mean)

sgdAccuracyMeanHashed = accuracy_score(sgdMeanHashed, y_test)
svmAccuracyMeanHashed = accuracy_score(svmMeanHashed, y_test)
rfAccuracyMeanHashed = accuracy_score(rfMeanHashed, y_test)
nbAccuracyMeanHashed = accuracy_score(nbMeanHashed, y_test)



sgdMaxHashed = TreeClassifiers.SGDClassify(x_train_max, y_train, x_test_max)
svmMaxHashed = TreeClassifiers.SVMClassifier(x_train_max, y_train, x_test_max)
rfMaxHashed = TreeClassifiers.rfClassify(x_train_max, y_train, x_test_max)
nbMaxHashed = TreeClassifiers.nbClassify(x_train_max, y_train, x_test_max)

sgdAccuracyMaxHashed = accuracy_score(sgdMaxHashed, y_test)
svmAccuracyMaxHashed = accuracy_score(svmMaxHashed, y_test)
rfAccuracyMaxHashed = accuracy_score(rfMaxHashed, y_test)
nbAccuracyMaxHashed = accuracy_score(nbMaxHashed, y_test)



sgdMinHashed = TreeClassifiers.SGDClassify(x_train_min, y_train, x_test_min)
svmMinHashed = TreeClassifiers.SVMClassifier(x_train_min, y_train, x_test_min)
rfMinHashed = TreeClassifiers.rfClassify(x_train_min, y_train, x_test_min)
nbMinHashed = TreeClassifiers.nbClassify(x_train_min, y_train, x_test_min)

sgdAccuracyMinHashed = accuracy_score(sgdMinHashed, y_test)
svmAccuracyMinHashed = accuracy_score(svmMinHashed, y_test)
rfAccuracyMinHashed = accuracy_score(rfMinHashed, y_test)
nbAccuracyMinHashed = accuracy_score(nbMinHashed, y_test)



sgdProdHashed = TreeClassifiers.SGDClassify(x_train_prod, y_train, x_test_prod)
svmProdHashed = TreeClassifiers.SVMClassifier(x_train_prod, y_train, x_test_prod)
rfProdHashed = TreeClassifiers.rfClassify(x_train_prod, y_train, x_test_prod)
nbProdHashed = TreeClassifiers.nbClassify(x_train_prod, y_train, x_test_prod)

sgdAccuracyProdHashed = accuracy_score(sgdProdHashed, y_test)
svmAccuracyProdHashed = accuracy_score(svmProdHashed, y_test)
rfAccuracyProdHashed = accuracy_score(rfProdHashed, y_test)
nbAccuracyProdHashed = accuracy_score(nbProdHashed, y_test)



print("Unsorted Sum Segmentation: \nSGD:", sgdAccuracyUSumHashed, "SVM:", svmAccuracyUSumHashed, 
"Random Forest", rfAccuracyUSumHashed, "NB:", nbAccuracyUSumHashed)
print("Sorted Sum Segmentation: \nSGD:", sgdAccuracySumHashed, "SVM:", svmAccuracySumHashed, 
"Random Forest", rfAccuracySumHashed, "NB:", nbAccuracySumHashed)
print()

print("Unsorted Mean Segmentation: \nSGD:", sgdAccuracyUMeanHashed, "SVM:", svmAccuracyUMeanHashed, 
"Random Forest", rfAccuracyUMeanHashed, "NB:", nbAccuracyUMeanHashed)
print("Sorted Mean Segmentation: \nSGD:",  sgdAccuracyMeanHashed, "SVM:", svmAccuracyMeanHashed, 
"Random Forest", rfAccuracyMeanHashed, "NB:", nbAccuracyMeanHashed)
print()

print("Unsorted Max Segmentation: \nSGD:", sgdAccuracyUMaxHashed, "SVM:", svmAccuracyUMaxHashed, 
"Random Forest", rfAccuracyUMaxHashed, "NB:", nbAccuracyUMaxHashed)
print("Sorted Max Segmentation: \nSGD:", sgdAccuracyMaxHashed, "SVM:", svmAccuracyMaxHashed, 
"Random Forest", rfAccuracyMaxHashed, "NB:", nbAccuracyMaxHashed)
print()

print("Unsorted Min Segmentation: \nSGD:", sgdAccuracyUMinHashed, "SVM:", svmAccuracyUMinHashed, 
"Random Forest", rfAccuracyUMinHashed, "NB:", nbAccuracyUMinHashed)
print("Sorted Min Segmentation: \nSGD:", sgdAccuracyMinHashed, "SVM:", svmAccuracyMinHashed, 
"Random Forest", rfAccuracyMinHashed, "NB:", nbAccuracyMinHashed)
print()

print("Unsorted Prod Segmentation: \nSGD:", sgdAccuracyUProdHashed, "SVM:", svmAccuracyUProdHashed, 
"Random Forest", rfAccuracyUProdHashed, "NB:", nbAccuracyUProdHashed)
print("Sorted Prod Segmentation: \nSGD:", sgdAccuracyProdHashed, "SVM:", svmAccuracyProdHashed, 
"Random Forest", rfAccuracyProdHashed, "NB:", nbAccuracyProdHashed)
print()







