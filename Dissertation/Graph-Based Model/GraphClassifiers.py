import random, math
import GraphDataProcessor as dp
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input
from statistics import mean, stdev
from sklearn.metrics import accuracy_score
from HiddenGraphLayer import HiddenGraphLayer as HGL
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier

def SGDClassify(x_train, y_train, x_test):
        text_clf = Pipeline([('clf', SGDClassifier())])
        text_clf.fit(x_train, y_train)

        predictions = text_clf.predict(x_test)
        return predictions

def SVMClassify(x_train, y_train, x_test):
    text_clf = Pipeline([('clf', LinearSVC())])
    text_clf.fit(x_train, y_train)

    predictions = text_clf.predict(x_test)
    return predictions

def rfClassify(x_train, y_train, x_test):
    """A RANDOM FOREST CLASSIFIER"""
    text_clf = Pipeline([('clf', RandomForestClassifier(n_estimators=5))])
    text_clf.fit(x_train, y_train)
    predictions = text_clf.predict(x_test)
    return predictions

def nbClassify(x_train, y_train, x_test):
        """A MULTINOMINAL NB CLASSIFIER"""
        # Build a pipeline to simplify the process of creating the vector matrix, transforming to tf-idf and classifying
        text_clf = Pipeline([('clf', MultinomialNB())])
        text_clf.fit(x_train, y_train)
        predictions = text_clf.predict(x_test)
        return predictions

def nbClassify(x_train, y_train, x_test):
        """A GAUSSIAN NB CLASSIFIER"""
        # Build a pipeline to simplify the process of creating the vector matrix, transforming to tf-idf and classifying
        text_clf = Pipeline([('clf', GaussianNB())])
        text_clf.fit(x_train, y_train)
        predictions = text_clf.predict(x_test)
        return predictions

class NBClassifier:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.crossValFolds = 5
        self.crossValFoldSize = int(len(self.x_train)/self.crossValFolds)

    def separationFunction(self):
        separatedDict = {}
        for i in range(len(self.x_train)):
            x = self.x_train[i]
            label = self.y_train[i]

            if label not in separatedDict:
                separatedDict[label] = []
            
            separatedDict[label].append(x)

        return separatedDict

    def splitIntoFolds(self):
        splitFolds, xTrain = [], self.x_train
        yFolds, yTrain= [], self.y_train
        for i in range(self.crossValFolds):
            splitFold, yFold = [], []
            while len(splitFold) < self.crossValFoldSize:
                index = random.randrange(len(xTrain))
                splitFold.append(xTrain.pop(index))
                yFold.append(yTrain.pop(index))
            splitFolds.append(splitFold)
            yFolds.append(yFold)
        return splitFolds, yFolds

    def calculateAccuracyScore(self, values, predictions):
        accuracyScore = 0
        for i in range(len(values)):
            if values[i] == predictions[i]:
                accuracyScore += 1
        
        accuracyScore = float(accuracyScore/len(values)) * 100.0

    def evaluate(self):
        splitFolds, yFolds = self.splitIntoFolds()
        accuracyScores = []

        i = 0
        for sf in splitFolds:
            sf = splitFolds[i]
            y = yFolds[i]
            xTrain, xTest = splitFolds, []
            xTrain.remove(sf)
            xTrain = sum(xTrain, [])
            for f in sf:
                thisX = f
                xTest.append(thisX)
                thisX[-1] = None
            predictions = self.runNBClassifier(xTest)
            accuracyScore = self.calculateAccuracyScore(y, predictions)
            accuracyScores.append(accuracyScore)

            i += 1
        return accuracyScores

    def getStandardDev(self, values):
        meanValue = sum(values)/float(len(values))
        stanDev = sum([(i-meanValue)**2 for i in values]) / float(len(values)-1)
        return math.sqrt(stanDev)

    def collectStatistics(self, xTrain):
        return [(sum(x)/float(len(x)), self.getStandardDev(x), len(x)) for x in zip(*xTrain)]

    def collectClassStatistics(self):
        separatedDict = self.separationFunction()
        classStats = {}
        for i, j in separatedDict.items():
            classStats[i] = self.collectStatistics(j)
        
        return classStats

    def getValProbability(self, val, valMean, valStd):
        if valStd == 0:
            valStd = 1
        e = 0.5
        if val is not None and val[-1] is not None:
            e = math.exp(((val-valMean)**2 / (2 * valStd**2))) 
            e =  e * (1 / (math.sqrt(2 * math.pi) * valStd))

        return e

    def getClassProbability(self, classStats, values):
        rowSum, probabilities = sum(classStats[y][0][2] for y in classStats), {}
        for value, stat in classStats.items():
            probabilities[value] = classStats[value][0][2]/float(rowSum)
            for i in range(len(stat)):
                valMean, valStd, valCount = stat[i]
                probabilities[value] *= self.getValProbability(values[i], valMean, valStd)

        return probabilities

    def makePrediction(self, classStats, values):
        probabilities = self.getClassProbability(classStats, values)
        label, probability = 10, -1
        for value, prob in probabilities.items():
            if label == 10 or prob > probability:
                probability = prob
                label = value
        
        return label

    def runNBClassifier(self, tester):
        classStats = self.collectClassStatistics()
        predictions = []
        prediction = self.makePrediction(classStats, tester)
        predictions.append(prediction)
    
        return predictions


xTrain1, yTrain1, xTest1, yTest1 = dp.runProcessor4()

nbc = NBClassifier(xTrain1, yTrain1, xTest1, yTest1)
summary = nbc.collectStatistics(xTrain1)
print(summary)
# separated = nbc.separationFunction()
# for label in separated:
# 	print(label)
# 	for row in separated[label]:
# 		print(row)

# scores = nbc.evaluate()
# print(scores)




# trainTestData1 = [xTrain1, yTrain1, xTest1, yTest1]

# hiddenSGD1 = HGL("sgd", trainTestData=trainTestData1)
# hiddenSVM1 = HGL("svm", trainTestData=trainTestData1)
# hiddenRF1 = HGL("rf", trainTestData=trainTestData1)
# hiddenNB1 = HGL("nb", trainTestData=trainTestData1)

# sgdPred1 = hiddenSGD1.chooseModel()
# sgdAccuracy1 = accuracy_score(sgdPred1, yTest1)

# svmPred1 = hiddenSVM1.chooseModel()
# svmAccuracy1 = accuracy_score(svmPred1, yTest1)

# rfPred1 = hiddenRF1.chooseModel()
# rfAccuracy1 = accuracy_score(rfPred1, yTest1)

# nbPred1 = hiddenNB1.chooseModel()
# nbAccuracy1 = accuracy_score(nbPred1, yTest1)



# xTrain2, yTrain2, xTest2, yTest2 = dp.runProcessor2()
# trainTestData2 = [xTrain2, yTrain2, xTest2, yTest2]
# hiddenSGD2 = HGL("sgd", trainTestData=trainTestData2)
# hiddenSVM2 = HGL("svm", trainTestData=trainTestData2)
# hiddenRF2 = HGL("rf", trainTestData=trainTestData2)
# hiddenNB2 = HGL("nb", trainTestData=trainTestData2)

# sgdPred2 = hiddenSGD2.chooseModel()
# sgdAccuracy2 = accuracy_score(sgdPred2, yTest2)

# svmPred2 = hiddenSVM2.chooseModel()
# svmAccuracy2 = accuracy_score(svmPred2, yTest2)

# rfPred2 = hiddenRF2.chooseModel()
# rfAccuracy2 = accuracy_score(rfPred2, yTest2)

# nbPred2 = hiddenNB2.chooseModel()
# nbAccuracy2 = accuracy_score(nbPred2, yTest2)



# print("NBC Segemented- ", nbAccuracy1) 
# print("NBC Padded - ", nbAccuracy2) 
# print()
# print("SGD Segemented - ", sgdAccuracy1) 
# print("SGD Padded - ", sgdAccuracy2) 
# print()
# print("SVC Segemented - ", svmAccuracy1)
# print("SVC Padded - ", svmAccuracy2)
# print()
# print("RF Segemented - ", rfAccuracy1)
# print("RF Padded - ", rfAccuracy2)







                
