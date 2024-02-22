import tensorflow as tf
import numpy as np
from tensorflow import keras

fashion_dataset = keras.datasets.fashion_mnist
# split the dataset
(training_img, training_labels), (test_img, test_labels) = fashion_dataset.load_data()

## model
model = keras.Sequential([
    keras.layers.Flatten(input_shape= (28, 28)), # the size of the input images 28 * 28
    keras.layers.Dense(units=128, activation= tf.nn.relu), # hidden layer has 128 neurons
    keras.layers.Dense(units=10, activation= tf.nn.softmax) # units = the size of the output labels, there are 10 labels
])
model.compile(optimizer= 'sgd', loss = 'mean_squared_error')
model.fit(training_img, training_labels, epochs= 1000)

model.evaluate(test_img, test_labels)
