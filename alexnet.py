import tensorflow as tf

from tensorflow.keras import Model, datasets, layers, models, initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt


def load_dataset():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)


def add_conv_layers(model, image_size):

    # Initialization: All layers are inittialized with normal dist. standard deviation=0.01 as said in report.
    # Padding: Not explained clear in report, I looked at a bunch of implementations and used to most common for each layer.
    # Number of filters, kernels sizes, stride sizes and pooling sizes are the same as in report.

    # 1st convolutional layer
    model.add(Conv2D(filters=96, input_shape=(image_size,image_size,3), kernel_size=(11,11), padding='valid', strides=(4,4), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)))
    # Max pooling of 1st layer
    model.add(MaxPooling2D(pool_size=(3,3), strides= (2,2), padding= 'valid'))

    # 2nd convolutional layer
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same', activation = 'relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)))
    # Max pooling of 2nd layer
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

    # 3rd convolutional layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides= 1, padding= 'same', activation= 'relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)))

    # 4th convolutional layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides= 1, padding= 'same', activation= 'relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)))

    # 5th convolutional layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides= 1, padding= 'same', activation= 'relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)))
    # Max pooling of 5th layer
    model.add(MaxPooling2D(pool_size=(3,3), strides= (2,2), padding= 'valid'))


load_dataset()

model = Sequential()
add_conv_layers(model, 32)
