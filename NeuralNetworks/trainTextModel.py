import pandas as pd
import RNN as NN
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# GET THE DATA/TEXT 
# with open("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Corona_NLP_train.csv") as file:
trainingData = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Corona_NLP_train.csv", encoding = "ISO-8859-1")
testingData = pd.read_csv("/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Corona_NLP_test.csv", encoding = "ISO-8859-1")


sentimentLabel = {'Extremely Negative': 0, 'Negative': 1, 'Neutral': 2, 'Positive': 3, 'Extremely Positive': 4}
trainingData['Sentiment'] =  [sentimentLabel[item] for item in trainingData['Sentiment']]
testingData['Sentiment'] =  [sentimentLabel[item] for item in testingData['Sentiment']]

# lines = [line.strip() for line in lines] 
# df1 = 
x_train = trainingData['OriginalTweet']
y_train = trainingData['Sentiment']

x_test = testingData['OriginalTweet']
y_test = testingData['Sentiment']

vectorizer = CountVectorizer(tokenizer=lambda doc: doc, min_df=2)
vectorizer.fit(x_train)

x_train = vectorizer.transform(x_train)
x_test  = vectorizer.transform(x_test)

print(x_train)
x_train = x_train.toarray()
x_test = x_test.toarray()
# x_train = np.reshape(x_train, (x_train.shape[0], len(x_train[0])))/255.
# x_test =  np.reshape(x_test, (x_test.shape[0], len(x_test[0])))/255.


epochs = 5
lr = 0.4
batch_size = 100
steps_per_epoch = 5

rnn  = NN.RNN([len(x_train), 128, 128, 5])
# rnn.info()
print('Steps per epoch:', steps_per_epoch)
history = rnn.train(
    x_train, y_train,
    x_test, y_test,
    epochs, steps_per_epoch,
    batch_size, lr
)