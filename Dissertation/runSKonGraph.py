from SKLearnClassifiers import SGDClassify, SVMClassify, rfClassify
from GraphDataProcessor import GraphDataProcessor

hashed = True
gdp = GraphDataProcessor(hashed)

# """RUNNING ON PADDED GRAPHS"""
# x_train, y_train, x_test, y_test = gdp.runProcessor2()

# sgdUSumPadAccuracy = SGDClassify(x_train, y_train, x_test, y_test)
# print("SGD CLASSIFIER AND PADDED GRAPHS:", sgdUSumPadAccuracy)

# rfUSumPadAccuracy = rfClassify(x_train, y_train, x_test, y_test)
# print("RANDOM FOREST CLASSIFIER AND PADDED GRAPHS:", rfUSumPadAccuracy)

# svmUSumPadAccuracy = SVMClassify(x_train, y_train, x_test, y_test)
# print("SVM CLASSIFIER AND PADDED GRAPHS:", svmUSumPadAccuracy)



"""RUNNING ON SEGMENTED GRAPHS"""
x_train, y_train, x_test, y_test = gdp.runProcessor4()

sgdUSumSegAccuracy = SGDClassify(x_train, y_train, x_test, y_test)
print("SGD CLASSIFIER AND SEGMENTED GRAPHS:", sgdUSumSegAccuracy)

rfUSumSegAccuracy = rfClassify(x_train, y_train, x_test, y_test)
print("RANDOM FOREST CLASSIFIER AND SEGMENTED GRAPHS:", rfUSumSegAccuracy)

svmUSumSegAccuracy = SVMClassify(x_train, y_train, x_test, y_test)
print("SVM CLASSIFIER AND SEGMENTED GRAPHS:", svmUSumSegAccuracy)

