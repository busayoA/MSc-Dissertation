import random, math
from statistics import mean

class NBClassifier:
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
        if valStd == 0:
            valStd = 1
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

    def getClassPriorProbability(self, yTrain: list):
        # P(y)
        numZeros = yTrain.count(0)
        numOnes = yTrain.count(1)
        totalValues = numZeros + numOnes

        priorProbZero = numZeros/totalValues
        priorProbOne = numOnes/totalValues

        return priorProbZero, priorProbOne

    def getEvidence(self, xValues: list):
        # P(X)
        xCount = 40 #number of values in each row = segmentCount
        evidenceProbs = []
        numXValues = len(xValues)
        for i in range(xCount):
            total = []
            evidenceProb = 0.0
            for j in xValues:
                current = j[i]
                total.append(current)
            overall = sum(total)
            evidenceProb = overall/numXValues
            evidenceProbs.append(evidenceProb)
        return math.prod(evidenceProbs)

    def getAllLikelihoods(self, xValues: list, yValues: list):
        # p(X|y)
        xCount = 40 #number of values in each row = segmentCount
        numZeros = yValues.count(0)
        numOnes = yValues.count(1)

        mergeProbs, quickProbs = [], []
        for i in range(xCount):
            mergeProb, quickProb = [], []
            for j in xValues:
                current = j[i]
                probXiIsMerge = current/numZeros
                probXiIsQuick = current/numOnes
                mergeProb.append(probXiIsMerge)
                quickProb.append(probXiIsQuick)
            mergeProbs.append(sum(mergeProb))
            quickProbs.append(sum(quickProb))
        return mergeProbs, quickProbs

    def makeMultinominalPredictions(self, x: list, mergeProbs: list, quickProbs: list, 
        priorProbMerge: float, priorProbQuick: float, evidence: float):
        predictions = []
        for i in range(len(x)):
            probXGivenMergeList, probXGivenQuickList = [], []
            for j in range(len(x[i])):
                probXGivenMerge = mergeProbs[j]
                probXGivenQuick = quickProbs[j]
                probXGivenMergeList.append(probXGivenMerge)
                probXGivenQuickList.append(probXGivenQuick)

            probMergeGivenX = (math.prod(probXGivenMergeList) * priorProbMerge)/evidence     
            probQuickGivenX = (math.prod(probXGivenQuickList) * priorProbQuick)/evidence 

            if probMergeGivenX > probQuickGivenX:
                predictions.append(0)
            else:
                predictions.append(1)
        return predictions

    def multinominalNBClassifier(self, xValues: list, yValues: list, xTest: list, yTest: list):
        # P(y|x0....x39)
        priorProbMerge, priorProbQuick = self.getClassPriorProbability(yValues)
        evidenceProbs = self.getEvidence(xValues)
        mergeProbs, quickProbs = self.getAllLikelihoods(xValues, yValues) 

        testPredictions = self.makeMultinominalPredictions(xTest, mergeProbs, quickProbs, priorProbMerge, 
            priorProbQuick, evidenceProbs)
        accuracyScore = self.calculateAccuracyScore(yTest, testPredictions)
        return accuracyScore

    def makePrediction(self, classStats, xValue):
        classProbabilities = self.getClassGaussianProbability(classStats, xValue)
        label, probability = 10, -1
        classProbs = classProbabilities.items()
        for value, prob in classProbs:
            if label == 10 or prob > probability:
                probability = prob
                label = value
        return label

    def runGNBClassifier(self, xTrain, yTrain, xTest):
        classStats = self.collateClassStatistics(xTrain, yTrain)
        predictions = []
        for x in xTest:
            prediction = self.makePrediction(classStats, x)
            predictions.append(prediction)
    
        return predictions

    def gaussianCrossValidation(self, xValues, yValues):
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
            
            predicted = self.runGNBClassifier(allTrain, allY, allTest)
            accuracyScore = self.calculateAccuracyScore(allTestY, predicted)
            accuracyScores.append(accuracyScore)
        return mean(accuracyScores)
