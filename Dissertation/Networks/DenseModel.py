import tensorflow as tf

def runDenseModel(x_train, y_train, x_test, y_test, activation: str, batchSize: int, epochs: int, filename: str = None):
    """ The method to run the model made up of densely connected layers
    x_train - The training data
    y_train - The training data labels
    x_test - The testing data
    y_test - The testing data labels
    activation: str -  The activation function to be applied
    batchSize: int - The size of each batch for the model
    epochs: int - The number of epochs 
    filename: str = None - THe name of the file to be saved """
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
        
