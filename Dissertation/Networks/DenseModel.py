import tensorflow as tf

def runDenseModel(x_train, y_train, x_test, y_test, activation: str, batchSize: int, epochs: int):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(x_train.shape[1], )))
    model.add(tf.keras.layers.Dense(128, activation=activation))
    model.add(tf.keras.layers.Dense(64, activation=activation))
    model.add(tf.keras.layers.Dense(2, activation=activation))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batchSize, validation_data=(x_test, y_test))

