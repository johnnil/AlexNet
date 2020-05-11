import numpy as np
import tensorflow as tf
import preprocessing

from tensorflow.keras import Model, datasets, layers, models, initializers, losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt


def add_conv_layers(model, image_size):

    # Initialization: All layers are inittialized with normal dist. standard deviation=0.01 as said in report.
    # Padding: Not explained clear in report, I looked at a bunch of implementations and used to most common for each layer.
    # Number of filters, kernels sizes, stride sizes and pooling sizes are the same as in report.

    # 1st convolutional layer
    model.add(Conv2D(filters=96, input_shape=(image_size,image_size,3), kernel_size=(11,11), strides=(4,4), padding='same', activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)))
    # Max pooling of 1st layer
    model.add(MaxPooling2D(pool_size=(3,3), strides= (2,2), padding='valid'))

    # 2nd convolutional layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation ='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)))
    # Max pooling of 2nd layer
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

    # 3rd convolutional layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides= 1, padding='same', activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)))

    # 4th convolutional layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides= 1, padding='same', activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)))

    # 5th convolutional layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides= 1, padding='same', activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)))
    # Max pooling of 5th layer

    if image_size != 32:
        # if image size is smaller than this pooling is impossible
        model.add(MaxPooling2D(pool_size=(3,3), strides= (2,2), padding='valid'))

def add_fully_connected(model, dropout=False):
    # "Unroll" 3D input to 1D
    model.add(Flatten())

    # 6th layer (fully connected)
    model.add(Dense(256, activation='relu'))
    if dropout:
        model.add(Dropout(0.4))

    # 7th layer (fully connected)
    model.add(layers.Dense(256, activation='relu'))
    if dropout:
        model.add(Dropout(0.4))

    # 8th layer (fully connected)
    model.add(layers.Dense(10)) # number of classes = 10
    if dropout:
        model.add(Dropout(0.4))




if __name__ == "__main__":
    # Build model
    model = Sequential()
    add_conv_layers(model, 32)
    add_fully_connected(model)
    model.summary()
    
    # Compile model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    # Load data
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Fit to data
    history = model.fit(train_images, train_labels, epochs=10,
                        batch_size=128, validation_split=0.05)

    # Plot results
    plt.plot(history.history['acc'], label='accuracy')
    plt.plot(history.history['val_acc'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print(test_acc)
