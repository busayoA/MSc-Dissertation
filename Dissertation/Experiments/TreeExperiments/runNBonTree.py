import TreeSegmentation as seg
import TreeDataProcessor as tdp
from NaiveBayes import NBClassifier

hashed = True
x_train_sum, x_train_mean, x_train_max, x_train_min, x_train_prod, y_train = seg.getSortedSegmentTrainData(hashed)
x_test_sum, x_test_mean, x_test_max, x_test_min, x_test_prod, y_test = seg.getSortedSegmentTestData(hashed)

x_train_sum = tdp.tensorToList(x_train_sum)
x_test_sum = tdp.tensorToList(x_test_sum)

y_train = tdp.floatToInt(y_train)
y_test = tdp.floatToInt(y_test)




nbc = NBClassifier(x, y)

print(nbc.multinominalNBClassifier(x_train_sum, y_train, x_test_sum, y_test))
