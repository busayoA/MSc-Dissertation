from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def SGDClassify(x_train, y_train, x_test):
    text_clf = Pipeline([('clf', SGDClassifier())])
    text_clf.fit(x_train, y_train)

    predictions = text_clf.predict(x_test)
    return predictions

def SVMClassifier(x_train, y_train, x_test):
    text_clf = Pipeline([('clf', LinearSVC())])
    text_clf.fit(x_train, y_train)

    predictions = text_clf.predict(x_test)
    return predictions

def rfClassify(x_train, y_train, x_test):
    """A RANDOM FOREST CLASSIFIER"""
    text_clf = Pipeline([('clf', RandomForestClassifier(n_estimators=5))])
    text_clf.fit(x_train, y_train)
    predictions = text_clf.predict(x_test)
    return predictions

def nbClassify(x_train, y_train, x_test):
    """A MULTINOMINAL NB CLASSIFIER"""
    # Build a pipeline to simplify the process of creating the vector matrix, transforming to tf-idf and classifying
    text_clf = Pipeline([('clf', MultinomialNB())])
    text_clf.fit(x_train, y_train)
    predictions = text_clf.predict(x_test)
    return predictions