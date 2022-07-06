import GraphDataProcessor as dp
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input
from sklearn.metrics import accuracy_score
from HiddenGraphLayer import HiddenGraphLayer as HGL

x_train, y_train, x_test, y_test = dp.runProcessor1()
optimizer = tf.keras.optimizers.RMSprop()
loss = tf.keras.losses.BinaryCrossentropy()
metrics = tf.keras.metrics.Accuracy()

dropout = HGL("dropout", dropoutRate=0.3)
outputLayer = HGL("output", activationFunction="sigmoid", neurons=2)

# HIDDEN LAYERS USING RELU ACTIVATION
# hiddenLSTM1 = HGL("lstm", learningRate=0.05, activationFunction="relu", neurons=64, dropoutRate=0.3)
# hiddenGRU1 = HGL("gru", learningRate=0.05, activationFunction="relu", neurons=64, dropoutRate=0.3)
# hiddenLSTMModel1 = Sequential()
# hiddenLSTMModel1.add(tf.keras.layers.InputLayer(input_shape=(x_train.shape[1], 1)))
# hiddenLSTMModel1.add(hiddenLSTM1.chooseModel(inputShape=(x_train.shape[1], 1), returnSequences=True))
# hiddenLSTMModel1.add(tf.keras.layers.LSTM(128))
# hiddenLSTMModel1.add(tf.keras.layers.Dense(2, activation='sigmoid'))
# hiddenLSTMModel1.summary()
# hiddenLSTMModel1.compile(loss=loss, optimizer=optimizer, metrics=metrics)
# hiddenLSTMModel1.fit(x_train, y_train, epochs=30, batch_size=10, validation_data=(x_test, y_test))

# hiddenGRUModel1 = Sequential()
# hiddenGRUModel1.add(tf.keras.layers.InputLayer(input_shape=(x_train.shape[1], 1)))
# hiddenGRUModel1.add(hiddenGRU1.chooseModel(inputShape=(x_train.shape[1], 1)))
# hiddenGRUModel1.add(tf.keras.layers.Dense(2, activation='sigmoid'))
# hiddenGRUModel1.summary()
# hiddenGRUModel1.compile(loss=loss, optimizer=optimizer, metrics=metrics)
# hiddenGRUModel1.fit(x_train, y_train, epochs=30, batch_size=10, validation_data=(x_test, y_test))




# HIDDEN LAYERS USING TANH ACTIVATION
hiddenLSTM2 = HGL("lstm", learningRate=0.05, activationFunction="tanh", neurons=64, dropoutRate=0.3)
hiddenGRU2 = HGL("gru", learningRate=0.05, activationFunction="tanh", neurons=64, dropoutRate=0.3)

hiddenLSTMModel2 = Sequential()
hiddenLSTMModel2.add(tf.keras.layers.InputLayer(input_shape=(x_train.shape[1], 1)))
hiddenLSTMModel2.add(hiddenLSTM2.chooseModel(inputShape=(x_train.shape[1], 1), returnSequences=True))
hiddenLSTMModel2.add(tf.keras.layers.LSTM(128))
hiddenLSTMModel2.add(tf.keras.layers.Dense(2, activation='sigmoid'))
hiddenLSTMModel2.summary()
hiddenLSTMModel2.compile(loss=loss, optimizer=optimizer, metrics=metrics)
hiddenLSTMModel2.fit(x_train, y_train, epochs=30, batch_size=10, validation_data=(x_test, y_test))

hiddenGRUModel2 = Sequential()
hiddenGRUModel2.add(tf.keras.layers.InputLayer(input_shape=(x_train.shape[1], 1)))
hiddenGRUModel2.add(hiddenGRU2.chooseModel(inputShape=(x_train.shape[1], 1)))
hiddenGRUModel2.add(tf.keras.layers.Dense(2, activation='sigmoid'))
hiddenGRUModel2.summary()
hiddenGRUModel2.compile(loss=loss, optimizer=optimizer, metrics=metrics)
hiddenGRUModel2.fit(x_train, y_train, epochs=30, batch_size=10, validation_data=(x_test, y_test))


# HIDDEN LAYERS USING SOFTMAX ACTIVATION
hiddenLSTM3 = HGL("lstm", learningRate=0.05, activationFunction="softmax", neurons=64, dropoutRate=0.3)
hiddenGRU3 = HGL("gru", learningRate=0.05, activationFunction="softmax", neurons=64, dropoutRate=0.3)

# HIDDEN LAYERS USING LOGSIGMOID ACTIVATION
hiddenLSTM4 = HGL("lstm", learningRate=0.05, activationFunction="logsigmoid", neurons=64, dropoutRate=0.3)
hiddenGRU4 = HGL("gru", learningRate=0.05, activationFunction="logsigmoid", neurons=64, dropoutRate=0.3)

# lstm1 = hiddenLSTM1.chooseModel(inputShape=(len(x_train[0]), ),  returnSequences=True)




