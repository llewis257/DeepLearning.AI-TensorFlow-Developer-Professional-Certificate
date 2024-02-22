import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_dataset = keras.datasets.fashion_mnist
# split the dataset
(training_img, training_labels), (test_img, test_labels) = fashion_dataset.load_data()


## normalize the image data - from 0-255 to 0-1
training_img = training_img/255
test_img = test_img/255

# implement a callback - to stop the model when we have desired output
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    accuracy = 95/100
    if(logs.get('accuracy') >= accuracy): # Experiment with changing this value
      print(f"\nReached {accuracy} accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
## model
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)), # the size of the input images 28 * 28
    keras.layers.Dense(units=128, activation= tf.nn.relu), # hidden layer has 128 neurons
    keras.layers.Dense(units=10, activation= tf.nn.softmax) # units = the size of the output labels, there are 10 labels
])
model.compile(optimizer= tf.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
print('training_img shape:', training_img.shape)
print('training_labels shape:', training_labels.shape)
model.fit(training_img, training_labels, epochs= 5, callbacks=[callbacks])

model.evaluate(test_img, test_labels)


print(test_labels[13])
print('Prediction:', np.argmax(model.predict(test_img[13:14])))