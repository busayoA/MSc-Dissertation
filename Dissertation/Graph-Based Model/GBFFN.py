import GraphDataProcessor as dp
import tensorflow as tf
import numpy as np
from statistics import mean
from keras.models import Sequential
from keras.layers import Input
from HiddenGraphLayer import HiddenGraphLayer as HGL


"""USING SEGMENTED GRAPHS"""
x_train, y_train, x_test, y_test = dp.runProcessor3()

dimensions = tf.shape(x_train[0], out_type=np.int32)
ffLayer1 = HGL("ffn", learningRate=0.003, activationFunction="relu", neurons=128, hiddenLayerCount=2, hiddenLayerUnits=[128, 128])
output1 = HGL("dense", neurons=2, activationFunction="sigmoid")
model1 = Sequential()
model1.add(Input(shape=(dimensions[0], )))
for i in range(2):
    model1.add(ffLayer1.chooseModel()[i])
model1.add(output1.chooseModel())
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model1.summary()

# graphs = np.reshape(graphs, (2, dimensions[1]))
history1 = model1.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

accuracy1 = mean(history1.history['accuracy'])




dimensions = tf.shape(x_train[0], out_type=np.int32)
ffLayer2 = HGL("ffn", learningRate=0.003, activationFunction="softmax", neurons=128, hiddenLayerCount=2, hiddenLayerUnits=[128, 128])
output2 = HGL("dense", neurons=2, activationFunction="sigmoid")
model2 = Sequential()
model2.add(Input(shape=(dimensions[0], )))
for i in range(2):
    model2.add(ffLayer2.chooseModel()[i])
model2.add(output2.chooseModel())
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model2.summary()

history2 = model2.fit(x_train, y_train, epochs=40, validation_data=(x_test, y_test))

accuracy2 = mean(history2.history['accuracy'])



dimensions = tf.shape(x_train[0], out_type=np.int32)
ffLayer3 = HGL("ffn", learningRate=0.003, activationFunction="tanh", neurons=128, hiddenLayerCount=2, hiddenLayerUnits=[128, 128])
output3 = HGL("dense", neurons=2, activationFunction="sigmoid")
model3 = Sequential()
model3.add(Input(shape=(dimensions[0], )))
for i in range(2):
    model3.add(ffLayer3.chooseModel()[i])
model3.add(output2.chooseModel())
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model3.summary()

history3 = model3.fit(x_train, y_train, epochs=40, validation_data=(x_test, y_test))

accuracy3 = mean(history3.history['accuracy'])


ffLayer4 = HGL("ffn", learningRate=0.003, activationFunction="logsigmoid", neurons=128, hiddenLayerCount=2, hiddenLayerUnits=[128, 128])
output4 = HGL("dense", neurons=2, activationFunction="sigmoid")
model4 = Sequential()
model4.add(Input(shape=(dimensions[0], )))
for i in range(2):
    model4.add(ffLayer4.chooseModel()[i])
model4.add(output2.chooseModel())
model4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model4.summary()
history4 = model4.fit(x_train, y_train, epochs=40, validation_data=(x_test, y_test))
accuracy4 = mean(history4.history['accuracy'])


"""USING PADDED GRAPHS"""
x_train, y_train, x_test, y_test = dp.runProcessor1()
dimensions = tf.shape(x_train[0], out_type=np.int32)
ffLayer5 = HGL("ffn", learningRate=0.003, activationFunction="relu", neurons=128, hiddenLayerCount=2, hiddenLayerUnits=[128, 128])
output5 = HGL("dense", neurons=2, activationFunction="sigmoid")
model5 = Sequential()
model5.add(Input(shape=(dimensions[0], )))
for i in range(2):
    model5.add(ffLayer5.chooseModel()[i])
model5.add(output5.chooseModel())
model5.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model5.summary()
history5 = model5.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
accuracy5 = mean(history5.history['accuracy'])




dimensions = tf.shape(x_train[0], out_type=np.int32)
ffLayer6 = HGL("ffn", learningRate=0.003, activationFunction="softmax", neurons=128, hiddenLayerCount=2, hiddenLayerUnits=[128, 128])
output6 = HGL("dense", neurons=2, activationFunction="sigmoid")
model6 = Sequential()
model6.add(Input(shape=(dimensions[0], )))
for i in range(2):
    model6.add(ffLayer6.chooseModel()[i])
model6.add(output6.chooseModel())
model6.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model6.summary()
history6 = model6.fit(x_train, y_train, epochs=40, validation_data=(x_test, y_test))
accuracy6 = mean(history6.history['accuracy'])



ffLayer7 = HGL("ffn", learningRate=0.003, activationFunction="tanh", neurons=128, hiddenLayerCount=2, hiddenLayerUnits=[128, 128])
output7 = HGL("dense", neurons=2, activationFunction="sigmoid")
model7 = Sequential()
model7.add(Input(shape=(dimensions[0], )))
for i in range(2):
    model7.add(ffLayer7.chooseModel()[i])
model7.add(output7.chooseModel())
model7.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model7.summary()
history7 = model7.fit(x_train, y_train, epochs=40, validation_data=(x_test, y_test))
accuracy7 = mean(history7.history['accuracy'])


ffLayer8 = HGL("ffn", learningRate=0.003, activationFunction="logsigmoid", neurons=128, hiddenLayerCount=2, hiddenLayerUnits=[128, 128])
output8 = HGL("dense", neurons=2, activationFunction="sigmoid")
model8 = Sequential()
model8.add(Input(shape=(dimensions[0], )))
for i in range(2):
    model8.add(ffLayer8.chooseModel()[i])
model8.add(output8.chooseModel())
model8.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model8.summary()
history8 = model8.fit(x_train, y_train, epochs=40, validation_data=(x_test, y_test))
accuracy8 = mean(history8.history['accuracy'])





# print("Unpadded with Relu:", accuracy1)
# print("Unpadded with Softmax:", accuracy2)
# print("Unpadded with Tanh:", accuracy3)
# print("Unpadded with Log Sigmoid", accuracy4)

print("Padded with Relu:", accuracy5)
print("Padded with Softmax:", accuracy5)
print("Padded with Tanh:", accuracy7)
print("Padded with Log Sigmoid", accuracy8)

