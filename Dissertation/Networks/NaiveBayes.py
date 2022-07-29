import random, math
from statistics import mean

class NBClassifier:
    def __init__(self, x: list, y: list):
        """A Naïve Bayes classifier model with cross-validation
        x: list - The x values (training data)
        y: list - The y values (training data labels)"""
        self.x = x
        self.y = y
        self.crossValFolds = 5 #the number of folds 
        self.crossValFoldSize = int(len(self.x)/self.crossValFolds) # the size of each fold

    def separationFunction(self, xValues: list, yValues: list):
        """ Separate the data based on classes to get a summary of the data
        xValues: list - The data 
        yValues: list - The data labels """
        separatedDict = {} #separate the data into a dictionary with the labels as the keys
        for i in range(len(xValues)):
            x = xValues[i]
            label = yValues[i]
            if label not in separatedDict:
                separatedDict[label] = []
            separatedDict[label].append(x)
        return separatedDict

    def splitIntoFolds(self, xValues, yValues):
        """Split the data and the labels into folds for cross-validation"""
        splitFolds, xVals = [], xValues #the folds for the data
        yFolds, yTrain = [], yValues # the folds for the labels
        for i in range(self.crossValFolds):
            currentFold, yFold = [], []
            while len(currentFold) < self.crossValFoldSize: #make sure the folds are all the same size
                j = random.randrange(len(xVals)) #randomly select data to be in each fold
                currentFold.append(xVals.pop(j))
                yFold.append(yTrain.pop(j)) #repeat for the label
            splitFolds.append(currentFold)
            yFolds.append(yFold)
        return splitFolds, yFolds

    def calculateAccuracyScore(self, values: list, predictions: list):
        """Get the overall accuracy score of the model
        values: list - the actual y values
        predictions: list - the predicted y values"""
        accuracyScore = 0.0
        for i in range(len(values)):
            if values[i] == predictions[i]:
                accuracyScore += 1.0
        
        accuracyScore = accuracyScore/float(len(values)) * 100.0
        return accuracyScore

    def getMean(self, values: list):
        """Get the mean of a set of values
        values: list - The lsit of values from which to calculate the mean"""
        return float(mean(values))

    def getStandardDev(self, values: list):
        """Calculate the standard deviation of a set of values 
        values: list - The lsit of values from which to calculate the standard deviation"""
        meanValue = self.getMean(values)
        var = sum([(i-meanValue) * (i-meanValue) for i in values]) / float(len(values)-1)
        return math.sqrt(var)

    def collateStatistics(self, values: list):
        """Get the statistics for a set of values
        values: list - The list from which the statistics are to be collected"""
        return [(self.getMean(i), self.getStandardDev(i), len(i)) for i in zip(*values)]

    def collateClassStatistics(self, xValues: list, yValues: list):
        """Get the statistics for an individual class
        xValues: list - The class data
        yValues: list - The class labels"""
        separatedDict = self.separationFunction(xValues, yValues) #separate the data by class
        classStats = {}

        # collate the statistics for each separate class
        for i, j in separatedDict.items():
            classStats[i] = self.collateStatistics(j)
        
        return classStats

    def getGaussianProbability(self, val, valMean, valStd):
        """Calculate the Gaussian probability for a set of values
        val - Each individual value
        valMean - The mean of the set of values from which 'val' came from
        valStd - The standard deviation of the set of values from which 'val' came from"""
        distFromMean = (val-valMean)  * (val-valMean) #the distance between the value and its mean
        if valStd == 0:
            valStd = 1 # if the standard deviation is 0, set it to 1 so it can be multiplied and divided

        # calculate the Gaussian probablilty using the above values
        stdSqr = 2 * valStd * valStd 
        e = distFromMean/stdSqr
        piSqrt = 1/(math.sqrt(2 * math.pi) * valStd)
        gaussianProb = math.exp(-(e)) * (piSqrt)

        return gaussianProb

    def getClassGaussianProbability(self, classStats, values):
        """Get the Gaussian probability of a class of values
        classStats - The statistics of the individual class
        values - The values in the class
        """
        rowSum, probabilities = sum(classStats[y][0][2] for y in classStats), {}
        for value, stat in classStats.items():
            probabilities[value] = classStats[value][0][2]/float(rowSum)
            for i in range(len(stat)):
                valMean, valStd, valCount = stat[i]
                # get the probability of a value belonging to the current class
                probabilities[value] *= self.getGaussianProbability(values[i], valMean, valStd) 

        return probabilities

    def makePrediction(self, classStats, xValue):
        """Make a prediction on a value in the testing fold based on its properties
        classStats - The statistics of all the classes
        xValue - The value who's class is to be predicted
        """
        # Get the gaussian probabilities of the class based on its mean and standard deviation
        classProbabilities = self.getClassGaussianProbability(classStats, xValue)
        label, probability = 10, -1
        classProbs = classProbabilities.items()
        for value, prob in classProbs:
            if label == 10 or prob > probability:
                probability = prob #select the highest probability as the final prediction
                label = value
        return label

    def runGNBClassifier(self, xTrain: list, yTrain: list, xTest: list):
        """Make classifications and predictions
        xTrain: list - The training data folds
        yTrain: list - The training label folds 
        xTest: list - The testing data fold"""
        # Get the mean and standard deviation for the training folds
        classStats = self.collateClassStatistics(xTrain, yTrain)
        predictions = []
        for x in xTest:
            # make the predictions on the testing data fold
            prediction = self.makePrediction(classStats, x)
            predictions.append(prediction)
    
        return predictions

    def gaussianCrossValidation(self, xValues: list, yValues: list):
        """Run the cross-validation algorithm on all the data
        xValues: list - The data
        yValues: list - The data class labels"""
        # split the data into folds of equal values:
        foldsX, foldsY = self.splitIntoFolds(xValues, yValues)
        accuracyScores = []

        # for each data fold, separate it from the rest of the folds 
        # use the other folds as training data and use the current fold as testing data
        # calculate the and return the mean accuracy score when the prediction process is complete.
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

