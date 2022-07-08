from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def SGDClassify(x_train, y_train, x_test, y_test):
    classifier = Pipeline([('clf', SGDClassifier())])
    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_test)
    accuracyScore = accuracy_score(y_test, predictions)
    return accuracyScore

def SVMClassify(x_train, y_train, x_test, y_test):
    classifier = Pipeline([('clf', LinearSVC())])
    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_test)
    accuracyScore = accuracy_score(y_test, predictions)
    return accuracyScore

def rfClassify(x_train, y_train, x_test, y_test):
    """A RANDOM FOREST CLASSIFIER"""
    classifier = Pipeline([('clf', RandomForestClassifier(n_estimators=5))])
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    accuracyScore = accuracy_score(y_test, predictions)
    return accuracyScore

