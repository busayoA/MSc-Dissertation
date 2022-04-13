import pandas as pd
import RNN as NN
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# GET THE DATA/TEXT 
# with open("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Corona_NLP_train.csv") as file:
trainingData = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Corona_NLP_train.csv", encoding = "ISO-8859-1")
testingData = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Corona_NLP_test.csv", encoding = "ISO-8859-1")
sentimentLabel = {'Extremely Negative': 0, 'Negative': 1, 'Neutral': 2, 'Positive': 3, 'Extremely Positive': 4}
trainingData['Sentiment'] =  [sentimentLabel[item] for item in trainingData['Sentiment']]
testingData['Sentiment'] =  [sentimentLabel[item] for item in testingData['Sentiment']]

# lines = [line.strip() for line in lines] 
# df1 = 
x = list(trainingData['OriginalTweet'])
y = list(trainingData['Sentiment'])

# x_test = list(testingData['OriginalTweet'])
# y_test = list(testingData['Sentiment'])

# print(lines)


def createEmbeddings(xData):
    embeddings = CountVectorizer(tokenizer=lambda doc: doc, min_df=2).fit_transform(xData)
    embeddings = embeddings.toarray()
    return embeddings

x = createEmbeddings(x)
x_train = x[:int(0.7*len(x))]
x_test = x[int(0.7*len(x)):]
x_train = np.reshape(x_train, (x_train.shape[0], len(x_train[0])))/255.
x_test =  np.reshape(x_test, (x_test.shape[0], len(x_test[0])))/255.


y_train = y[:int(0.7*len(y))]
y_test = y[int(0.7*len(y)):]

print(len(y_train))
print(len(y_test))

epochs = 5
lr = 0.4
batch_size = 100
steps_per_epoch = 50

rnn  = NN.RNN([len(x_train), 100, 100, 5])
rnn.info()
print('Steps per epoch:', steps_per_epoch)
history = rnn.train(
    x_train, y_train,
    x_test, y_test,
    epochs, steps_per_epoch,
    batch_size, lr
)
# # rnn.trainModel(x_train, y_train, x_test, y_test)
# # for i in range(len(x_train[:60])):
# #     prediction = rnn.predict(x_train[i])
# #     print("Expected=", y_train[i], " predicted", prediction[0][0])

# # for i in range(len(x_test)):
# #     prediction2 = rnn.predict(x_test[i])
# #     print("Expected=", y_test[i], " predicted", prediction2[0][0])