import numpy as np
import tensorflow as tf
from tensorflow import keras

def neural_net(pred: int) -> int:
    layers = [
        keras.layers.Dense(units=1, input_shape=[1])
    ] 
    # defining the layers of the model 
    # -> 1 index = 1 layer
    # -> units=1  = 1 neuron
    model = tf.keras.Sequential(layers=layers)
    model.compile(optimizer= 'sgd', loss = 'mean_squared_error')
    print(type(model))

    # input training data
    x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=int)
    # expected output training data
    y = np.array([-7, -5, -1, -1, 1, 3, 5], dtype=int)
    # expected formula Y = 2*X - 1
    #training
    model.fit(x,y,epochs=100)
    prediction = model.predict([pred])
    return int(round(float(prediction), ndigits=0))

if __name__ == '__main__':
    print(neural_net(12))