import tensorflow as tf
import numpy as np
import sys

if len(sys.argv) != 2:
    print('Wrong number of arguments!')
    exit()

# Set the same seed
np.random.seed(424242)

model = tf.keras.models.load_model(sys.argv[1])
model.summary()

_, (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
test_images = test_images / 255.0

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)