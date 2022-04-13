import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import RNN as NN
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    
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

x_train = x_train.toarray()
x_test = x_test.toarray()

batchSize = 64
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

print([layer.supports_masking for layer in model.layers])

predictions = model.predict(np.array([x_train[:1]]))
print(predictions[0])
