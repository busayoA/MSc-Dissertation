import tensorflow as tf

def runDenseModel(x_train, y_train, x_test, y_test, activation: str, batchSize: int, epochs: int, filename: str = None):
    model = tf.keras.models.Sequential()
    #Insert the input layer
    model.add(tf.keras.layers.InputLayer(input_shape=(x_train.shape[1], ))) 

    # Add the densely connected layers
    model.add(tf.keras.layers.Dense(256, activation=activation))
    model.add(tf.keras.layers.Dense(64, activation=activation))
    model.add(tf.keras.layers.Dense(2, activation=activation))

    # Compile and train/fit the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batchSize, validation_data=(x_test, y_test))
    if filename is not None:
        model.save(filename)
        
