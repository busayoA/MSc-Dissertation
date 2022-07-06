import GraphDataProcessor as dp
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input
from sklearn.metrics import accuracy_score
from HiddenGraphLayer import HiddenGraphLayer as HGL

xTrain1, yTrain1, xTest1, yTest1 = dp.runProcessor4()

trainTestData1 = [xTrain1, yTrain1, xTest1, yTest1]

hiddenSGD1 = HGL("sgd", trainTestData=trainTestData1)
hiddenSVM1 = HGL("svm", trainTestData=trainTestData1)
hiddenRF1 = HGL("rf", trainTestData=trainTestData1)
hiddenNB1 = HGL("nb", trainTestData=trainTestData1)

sgdPred1 = hiddenSGD1.chooseModel()
sgdAccuracy1 = accuracy_score(sgdPred1, yTest1)

svmPred1 = hiddenSVM1.chooseModel()
svmAccuracy1 = accuracy_score(svmPred1, yTest1)

rfPred1 = hiddenRF1.chooseModel()
rfAccuracy1 = accuracy_score(rfPred1, yTest1)

nbPred1 = hiddenNB1.chooseModel()
nbAccuracy1 = accuracy_score(nbPred1, yTest1)



xTrain2, yTrain2, xTest2, yTest2 = dp.runProcessor2()
trainTestData2 = [xTrain2, yTrain2, xTest2, yTest2]
hiddenSGD2 = HGL("sgd", trainTestData=trainTestData2)
hiddenSVM2 = HGL("svm", trainTestData=trainTestData2)
hiddenRF2 = HGL("rf", trainTestData=trainTestData2)
hiddenNB2 = HGL("nb", trainTestData=trainTestData2)

sgdPred2 = hiddenSGD2.chooseModel()
sgdAccuracy2 = accuracy_score(sgdPred2, yTest2)

svmPred2 = hiddenSVM2.chooseModel()
svmAccuracy2 = accuracy_score(svmPred2, yTest2)

rfPred2 = hiddenRF2.chooseModel()
rfAccuracy2 = accuracy_score(rfPred2, yTest2)

nbPred2 = hiddenNB2.chooseModel()
nbAccuracy2 = accuracy_score(nbPred2, yTest2)



print("NBC Segemented- ", nbAccuracy1) 
print("NBC Padded - ", nbAccuracy2) 
print()
print("SGD Segemented - ", sgdAccuracy1) 
print("SGD Padded - ", sgdAccuracy2) 
print()
print("SVC Segemented - ", svmAccuracy1)
print("SVC Padded - ", svmAccuracy2)
print()
print("RF Segemented - ", rfAccuracy1)
print("RF Padded - ", rfAccuracy2)