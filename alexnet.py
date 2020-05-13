import numpy as np
import tensorflow as tf
import preprocessing

from tensorflow.keras import Model, datasets, layers, models, initializers, losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, ReLU
import matplotlib.pyplot as plt


def add_conv_layers(model, image_size):

    # Initialization: All layers are inittialized with normal dist. standard deviation=0.01 as said in report.
    # Padding: Not explained clear in report, I looked at a bunch of implementations and used to most common for each layer.
    # Number of filters, kernels sizes, stride sizes and pooling sizes are the same as in report.

    # 1st convolutional layer
    model.add(Conv2D(filters=64, input_shape=(image_size,image_size,3), kernel_size=(3,3), padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.01)))
    # Batch normalisation before ReLU as done in SimpleNet
    model.add(BatchNormalization())
    # ReLU
    model.add(ReLU())
    # Max pooling of 1st layer
    model.add(MaxPooling2D(pool_size=(3,3), strides= (2,2), padding='valid'))

    # 2nd convolutional layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.01)))
    # Batch normalisation before ReLU as done in SimpleNet
    model.add(BatchNormalization())
    # ReLU
    model.add(ReLU())
    # Max pooling of 2nd layer
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

    # 3rd convolutional layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.01)))
    # Batch normalisation before ReLU as done in SimpleNet
    model.add(BatchNormalization())
    # ReLU
    model.add(ReLU())

    # 4th convolutional layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.01)))
    # Batch normalisation before ReLU as done in SimpleNet
    model.add(BatchNormalization())
    # ReLU
    model.add(ReLU())

    # 5th convolutional layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', kernel_initializer=initializers.RandomNormal(stddev=0.01)))
    # Batch normalisation before ReLU as done in SimpleNet
    model.add(BatchNormalization())
    # ReLU
    model.add(ReLU())
    
    # Max pooling of 5th layer
    # if image size is smaller than this pooling is impossible
    model.add(MaxPooling2D(pool_size=(3,3), strides= (2,2), padding='valid'))

def add_fully_connected(model):
    # "Unroll" 3D input to 1D
    model.add(Flatten())

    # 6th layer (fully connected)
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    # 7th layer (fully connected)
    model.add(layers.Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    # 8th layer (fully connected), softmax activation
    model.add(layers.Dense(10)) # number of classes = 10


if __name__ == "__main__":
    # Set the seed
    np.random.seed(424242)

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

    # Duplicate data
    flipped_images = preprocessing.flip_data(train_images)

    # Verify data
    preprocessing.verify_data(flipped_images, train_labels)
    preprocessing.verify_data(train_images, train_labels)

    # Stack data
    train_images = np.vstack((train_images, flipped_images))
    train_labels = np.vstack((train_labels, train_labels))

    # Fit to data
    history = model.fit(train_images, train_labels, epochs=20,
                        batch_size=32, validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    # Save model
    model.save('new_model.h5')

    # Plot results
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    
