from sklearn.metrics import accuracy_score
from SKLearnClassifiers import SGDClassify, SVMClassify, rfClassify
import TreeSegmentation as seg
import TreeDataProcessor as tdp


""" WARNING!!!! IF HASHED IS TRUE, THEN YOU ARE USING HASHED DATA, IF IT IS FALSE, YOU ARE USING UNHASHED DATA """

hashed = True
x_train_usum, x_train_umean, x_train_umax, x_train_umin, x_train_uprod, y_train = seg.getUnsortedSegmentTrainData(hashed)
x_test_usum, x_test_umean, x_test_umax, x_test_umin, x_test_uprod, y_test = seg.getUnsortedSegmentTestData(hashed)

x_train_sum, x_train_mean, x_train_max, x_train_min, x_train_prod, y_train = seg.getSortedSegmentTrainData(hashed)
x_test_sum, x_test_mean, x_test_max, x_test_min, x_test_prod, y_test = seg.getSortedSegmentTestData(hashed)






