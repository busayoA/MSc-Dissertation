import nltk, string, csv
import pandas as pd
# GET THE DATA/TEXT 
# with open("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Corona_NLP_train.csv") as file:
trainingData = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Corona_NLP_train.csv", encoding = "ISO-8859-1")
testingData = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Corona_NLP_test.csv", encoding = "ISO-8859-1")

sentimentLabel = {'Extremely Negative': 0, 'Negative': 1, 'Neutral': 2, 'Positive': 3, 'Extremely Positive': 4}
trainingData['Sentiment'] =  [sentimentLabel[item] for item in trainingData['Sentiment']]
testingData['Sentiment'] =  [sentimentLabel[item] for item in testingData['Sentiment']]

trainX = trainingData['OriginalTweet']
trainY = trainingData['Sentiment']

testX = testingData['OriginalTweet']
testY = testingData['Sentiment']

finalY = []
def cleanData(xData, yData, fileToWriteX, fileToWriteY):
    y = []
    with open(fileToWriteX, 'w', encoding='UTF8') as file:
        writer = csv.writer(file)
        writer.writerow(['tweet'])
        for i in range(len(xData)):
            xData[i] = xData[i].lower()
            text = "".join([char for char in xData[i] if char not in string.punctuation])
            text = nltk.word_tokenize(text)
            if len(text) > 10:
                y.append(yData[i])
                text = " ".join([char for char in text if not char.startswith('http')])
                print(text)
                writer.writerow([text])
    
    with open(fileToWriteY, 'w', encoding='UTF8') as file:
        writer = csv.writer(file)
        writer.writerow(['tweet'])
        for i in range(len(y)):
            writer.writerow([y[i]])

    return xData, y

# train, yTrain = cleanData(trainX, trainY, 'cleanTrain.csv', 'yTrain.csv')
# test, yTest = cleanData(testX, testY, 'cleanTest.csv', 'yTest.csv')

xTrain = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/cleanTrain.csv", encoding = "UTF8")
xTest = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/cleanTest.csv", encoding = "UTF8")

yTrain = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/yTrain.csv", encoding = "UTF8")
yTest = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/yTest.csv", encoding = "UTF8")

xTrain = list(xTrain['tweet'])
xTest = list(xTest['tweet'])
yTrain = list(yTrain['tweet'])
yTest = list(yTest['tweet'])

# print(len(xTrain), len(yTrain), len(xTest), len(yTest))

def getData():
    return xTrain, yTrain, xTest, yTest


# data = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/fifa.csv", encoding = "UTF8")
# print(data.columns)
# data = data[["Position", 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
#        'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
#        'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
#        'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
#        'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
#        'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
#        'GKKicking', 'GKPositioning', 'GKReflexes']]
# print(data.columns)

# forward_player = ["ST", "LW", "RW", "LF", "RF", "RS","LS", "CF"]
# midfielder_player = ["CM","RCM","LCM", "CDM","RDM","LDM", "CAM", "LAM", "RAM", "RM", "LM"]
# defender_player = ["CB", "RCB", "LCB", "LWB", "RWB", "LB", "RB"]

# data.loc[data["Position"] == "GK", "Position"] = 0
# data.loc[data["Position"].isin(defender_player), "Position"] = 1
# data.loc[data["Position"].isin(midfielder_player), "Position"] = 2
# data.loc[data["Position"].isin(forward_player), "Position"] = 3


