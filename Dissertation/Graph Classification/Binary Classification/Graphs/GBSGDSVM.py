from BinaryGraphInputLayer import BinaryGraphInputLayer as BGIL
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from os.path import dirname, join


bgil = BGIL()
# x_train_nodes, x_train_matrix, y_train, x_test_nodes, x_test_matrix, y_test = bgil.readFiles()
# x_train = bgil.prepareData(x_train_nodes, x_train_matrix)

def readFiles():
    current_dir = dirname(__file__)

    merge = "./Data/Merge Sort"
    quick = "./Data/Quick Sort"

    merge = join(current_dir, merge)
    quick = join(current_dir, quick)

    mergeGraphs, mergeLabels = bgil.assignLabels(merge) 
    quickGraphs, quickLabels = bgil.assignLabels(quick) 

    mergeSplit = int(0.6 * len(mergeGraphs))
    quickSplit = int(0.6 * len(quickGraphs))

    x_train = mergeGraphs[:mergeSplit] + quickGraphs[:quickSplit]
    y_train = mergeLabels[:mergeSplit] + quickLabels[:quickSplit]

    print(y_train)
    # xTrain, yTrain, xTest, yTest = bgil.splitTrainTest(merge, quick)
    # x_train_nodes, x_train_matrix, y_train, x_test_nodes, x_test_matrix, y_test = self.getDatasets(xTrain, yTrain, xTest, yTest)

    # return x_train_nodes, x_train_matrix, y_train, x_test_nodes, x_test_matrix, y_test


readFiles()

def SGD(trainSet, trainCategories, testSet):
    text_clf = Pipeline([('clf', SGDClassifier())])
    text_clf.fit(trainSet, trainCategories)

    predictions = text_clf.predict(testSet)
    return predictions

# x = SGD(x_train_matrix, y_train, x_test_matrix)

