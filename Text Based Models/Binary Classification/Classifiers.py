import readFiles as rf
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def nbClassify(trainSet, trainCategories, testSet):
    """A MULTINOMINAL NB CLASSIFIER"""
    # Build a pipeline to simplify the process of creating the vector matrix, transforming to tf-idf and classifying
    text_clf = Pipeline([('vect', CountVectorizer(min_df=5)),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
    text_clf.fit(trainSet, trainCategories)
    predictions = text_clf.predict(testSet)
    return predictions

def sgdClassify(trainSet, trainCategories, testSet):
    """A LINEAR MODEL WITH STOCHASTIC GRADIENT DESCENT"""
    text_clf = Pipeline([('vect', CountVectorizer(min_df=5)),('tfidf', TfidfTransformer()),('clf', SGDClassifier())])
    text_clf.fit(trainSet, trainCategories)
    predictions = text_clf.predict(testSet)
    return predictions

def svcClassify(trainSet, trainCategories, testSet):
    """A SUPPORT VECTOR MACHINE CLASSIFIER"""
    text_clf = Pipeline([('vect', CountVectorizer(min_df=5)),('tfidf', TfidfTransformer()),('clf', LinearSVC())])
    text_clf.fit(trainSet, trainCategories)
    predictions = text_clf.predict(testSet)
    return predictions

def rfClassify(trainSet, trainCategories, testSet):
    """A RANDOM FOREST CLASSIFIER"""
    text_clf = Pipeline([('vect', CountVectorizer(min_df=5)),('tfidf', TfidfTransformer()),('clf', RandomForestClassifier(n_estimators=5))])
    text_clf.fit(trainSet, trainCategories)
    predictions = text_clf.predict(testSet)
    return predictions


# # Test on code
x_train, y_train, x_test, y_test = rf.getCodeData()
nbPred = nbClassify(x_train, y_train, x_test)
nbAccuracy = accuracy_score(nbPred, y_test)

sgdPred = sgdClassify(x_train, y_train, x_test)
sgdAccuracy = accuracy_score(sgdPred, y_test)

svcPred = svcClassify(x_train, y_train, x_test)
svcAccuracy = accuracy_score(svcPred, y_test)

rfPred = nbClassify(x_train, y_train, x_test)
rfAccuracy = accuracy_score(rfPred, y_test)

print("NBC - ", nbAccuracy) 
print("SGD - ", sgdAccuracy) 
print("SVC - ", svcAccuracy)
print("RF - ", rfAccuracy)
