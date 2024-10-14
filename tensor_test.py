import os

import tensorflow as tf
from tensorflow import keras

import tf2onnx

print(tf.version.VERSION)

model = keras.models.load_model('Model/pretrainedResnet.h5')
model.summary()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D

for l in model.layers:
    if isinstance(l, Conv2D):
        print(l.name, l.weights)
    elif isinstance(l, Dense):
        print(l.name, l.weights)

