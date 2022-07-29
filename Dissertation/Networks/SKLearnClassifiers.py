from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def SGDClassify(x_train: list, y_train: list, x_test: list, y_test: list):
    """A Stochastic Gradient Classifier
    x_train: list - The training data
    y_train: list - The training data labels
    x_test: list - The testing data 
    y_test: list - The testing data labels
    """
    classifier = Pipeline([('clf', SGDClassifier())]) #create a classification pipeline
    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_test)
    accuracyScore = accuracy_score(y_test, predictions)
    return accuracyScore

def SVMClassify(x_train: list, y_train: list, x_test: list, y_test: list):
    """A Support Vector Machine Classifier
    x_train: list - The training data
    y_train: list - The training data labels
    x_test: list - The testing data 
    y_test: list - The testing data labels
    """
    classifier = Pipeline([('clf', LinearSVC())])
    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_test)
    accuracyScore = accuracy_score(y_test, predictions)
    return accuracyScore

def rfClassify(x_train: list, y_train: list, x_test: list, y_test: list):
    """A RANDOM FOREST CLASSIFIER
    x_train: list - The training data
    y_train: list - The training data labels
    x_test: list - The testing data 
    y_test: list - The testing data labels
    """
    classifier = Pipeline([('clf', RandomForestClassifier())])
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    accuracyScore = accuracy_score(y_test, predictions)
    return accuracyScore

