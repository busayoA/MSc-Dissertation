#Converts the source code files to txt files for processing
import os

bubbleTrain = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Training Data/Bubble Sort"
insertionTrain = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Training Data/Insertion Sort"
mergeTrain = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Training Data/Merge Sort"
quickTrain = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Training Data/Quick Sort"
selectionTrain = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Training Data/Selection Sort"

 
bubbleTest = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Testing Data/Bubble Sort"
insertionTest = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Testing Data/Insertion Sort"
mergeTest = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Testing Data/Merge Sort"
quickTest = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Testing Data/Quick Sort"
selectionTest = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Testing Data/Selection Sort"

#className: Bubble Sort = 0
#className: Insertion Sort = 1
#className: Merge Sort = 2
#className: Quick Sort = 3
#className: Selection Sort = 4


def readFile(filePath):
    with open(filePath, 'r') as f:
        return f.read()

def assignLabels(filePath, fileList, labelList):
    os.chdir(filePath)
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".py"):
            path = f"{filePath}/{file}"
            # call read text file function
            fileList.append(readFile(path))
            if path.find("Bubble") != -1:
                labelList.append(0)
            elif path.find("Insertion") != -1:
                labelList.append(1)
            elif filePath.find("Merge") != -1:
                labelList.append(2)
            elif filePath.find("Quick") != -1:
                labelList.append(3)
            elif filePath.find("Selection") != -1:
                labelList.append(4)

def createTrainTestData():
    trainingFiles, trainingLabels, testingFiles, testingLabels = [], [], [], [] #the training and testing data

    # create training data:
    assignLabels(bubbleTrain, trainingFiles, trainingLabels)
    assignLabels(insertionTrain, trainingFiles, trainingLabels)
    assignLabels(mergeTrain, trainingFiles, trainingLabels)
    assignLabels(quickTrain, trainingFiles, trainingLabels)
    assignLabels(selectionTrain, trainingFiles, trainingLabels)

    # create testing data:
    assignLabels(bubbleTest, testingFiles, testingLabels)
    assignLabels(insertionTest, testingFiles, testingLabels)
    assignLabels(mergeTest, testingFiles, testingLabels)
    assignLabels(quickTest, testingFiles, testingLabels)
    assignLabels(selectionTest, testingFiles, testingLabels)

    return trainingFiles, trainingLabels, testingFiles, testingLabels
# [print(i, end="\n") for i in testingFiles]
# print(testingLabels)




