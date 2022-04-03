from cv2 import sort
from torch import empty
from sklearn.feature_extraction.text import CountVectorizer

# vectorizer = CountVectorizer(min_df=20)

def createEmbeddings(xData):

    embeddings = CountVectorizer(tokenizer=lambda doc: doc, min_df=2).fit_transform(xData)
    embeddings = embeddings.toarray()

    return embeddings


# rnn = Network.RecurrentNeuralNetwork(20, 0.3, 120, [60, 128, 128, 5]) 

# measures = rnn.trainModel(x_train, y_train, x_test, y_test)


