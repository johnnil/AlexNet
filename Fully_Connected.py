import tensorflow as tf
from tensorflow.keras import layers

"""
This class adds dense/fully connected layers to a CNN model
:param: model: a CNN model 
"""

def add_dense_layers(model):
    # "Unroll" 3D input to 1D
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    # 10 outputs = one for each class
    model.add(layers.Dense(10))
