import tensorflow as tf
from tensorflow.keras import Model, Mayers, Conv2D, MaxPooling2D;

"""
NOT DONE
This class adds 5 convolutional layers to a modell
:param: model: a CNN model
"""

def add_conv_layers(model):

    # 1st layer
    model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), padding='valid' strides=(4,4), activation='relu'))
    # Max pooling of 1st layer
    model.add(MaxPooling2D(pool_size=(3,3), strides= (2,2), padding= 'valid', data_format= None))

    # 2nd layer

    # Max pooling of 2nd layer
