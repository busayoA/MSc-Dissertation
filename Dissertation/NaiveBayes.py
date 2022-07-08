import random, math
import TreeSegmentation as seg
import TreeDataProcessor as tdp
from statistics import mean

class GaussianNBClassifier:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.crossValFolds = 5
        self.crossValFoldSize = int(len(self.x)/self.crossValFolds)

    def separationFunction(self, xValues, yValues):
        separatedDict = {}
        for i in range(len(xValues)):
            x = xValues[i]
            label = yValues[i]
            if label not in separatedDict:
                separatedDict[label] = []
            separatedDict[label].append(x)

        return separatedDict

    def splitIntoFolds(self, xValues, yValues):
        splitFolds, xVals = [], xValues
        yFolds, yTrain = [], yValues
        for i in range(self.crossValFolds):
            currentFold, yFold = [], []
            while len(currentFold) < self.crossValFoldSize:
                j = random.randrange(len(xVals))
                currentFold.append(xVals.pop(j))
                yFold.append(yTrain.pop(j))
            splitFolds.append(currentFold)
            yFolds.append(yFold)
        return splitFolds, yFolds

    def calculateAccuracyScore(self, values, predictions):
        accuracyScore = 0.0
        for i in range(len(values)):
            if values[i] == predictions[i]:
                accuracyScore += 1.0
        
        accuracyScore = accuracyScore/float(len(values)) * 100.0
        return accuracyScore

    def getMean(self, values):
        return float(mean(values))

    def getStandardDev(self, values):
        meanValue = self.getMean(values)
        var = sum([(i-meanValue) * (i-meanValue) for i in values]) / float(len(values)-1)
        return math.sqrt(var)

    def collateStatistics(self, values):
        return [(self.getMean(i), self.getStandardDev(i), len(i)) for i in zip(*values)]

    def collateClassStatistics(self, xValues, yValues):
        separatedDict = self.separationFunction(xValues, yValues)
        classStats = {}
        for i, j in separatedDict.items():
            classStats[i] = self.collateStatistics(j)
        
        return classStats

    def getGaussianProbability(self, val, valMean, valStd):
        distFromMean = (val-valMean)  * (val-valMean) 
        stdSqr = 2 * valStd * valStd 
        e = distFromMean/stdSqr
        piSqrt = 1/(math.sqrt(2 * math.pi) * valStd)
        gaussianProb = math.exp(-(e)) * (piSqrt)

        return gaussianProb

    def getClassGaussianProbability(self, classStats, values):
        rowSum, probabilities = sum(classStats[y][0][2] for y in classStats), {}
        for value, stat in classStats.items():
            probabilities[value] = classStats[value][0][2]/float(rowSum)
            for i in range(len(stat)):
                valMean, valStd, valCount = stat[i]
                probabilities[value] *= self.getGaussianProbability(values[i], valMean, valStd)

        return probabilities
 
    def makePrediction(self, classStats, xValue):
        classProbabilities = self.getClassGaussianProbability(classStats, xValue)
        label, probability = 10, -1
        classProbs = classProbabilities.items()
        for value, prob in classProbs:
            if label == 10 or prob > probability:
                probability = prob
                label = value
        return label

    def runNBClassifier(self, xTrain, yTrain, xTest):
        classStats = self.collateClassStatistics(xTrain, yTrain)
        predictions = []
        for x in xTest:
            prediction = self.makePrediction(classStats, x)
            predictions.append(prediction)
    
        return predictions

    def crossValidation(self, xValues, yValues):
        foldsX, foldsY = self.splitIntoFolds(xValues, yValues)
        accuracyScores = []
        for i in range(len(foldsX)):
            currentTrainX, currentY = foldsX[i], foldsY[i]
            allTrain, allY, allTest, allTestY = list(foldsX), list(foldsY), [], []

            allY.remove(currentY)
            allTrain.remove(currentTrainX)
            allTrain = sum(allTrain, [])
            allY = sum(allY, [])
            for j in range(len(currentTrainX)):
                trainY = currentY[j]
                trainX = currentTrainX[j]
                allTest.append(trainX)
                allTestY.append(trainY)
            
            predicted = self.runNBClassifier(allTrain, allY, allTest)
            accuracyScore = self.calculateAccuracyScore(allTestY, predicted)
            accuracyScores.append(accuracyScore)
        return mean(accuracyScores)

hashed = False
x_train_sum, x_train_mean, x_train_max, x_train_min, x_train_prod, y_train = seg.getSortedSegmentTrainData(hashed)
x_test_sum, x_test_mean, x_test_max, x_test_min, x_test_prod, y_test = seg.getSortedSegmentTestData(hashed)

x_train_sum = tdp.tensorToList(x_train_sum)
x_test_sum = tdp.tensorToList(x_test_sum)

y_train = tdp.floatToInt(y_train)
y_test = tdp.floatToInt(y_test)


x = []
for i in x_train_sum:
    x.append(i)
for i in x_test_sum:
    x.append(i)

y = []
for i in y_train:
    y.append(i)
for i in y_test:
    y.append(i)
# x = x_train_sum + x_test_sum
# y = y_train + y_test
nbc = GaussianNBClassifier(x, y)
# # summary = nbc.collateClassStatistics(x_train_sum, y_train)
# # labels = nbc.runNBClassifier(x_test_sum, y_test)


# # [print(y_test[i], labels[i]) for i in range(len(labels))]

print(nbc.crossValidation(x, y))
print()