import numpy as np
import pickle
import scipy.misc as sp
import matplotlib.pyplot as plt
import tensorflow as tf

def LoadBatch(filename):
	""" Copied from the dataset website """
	with open('Dataset/'+filename, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict



def load_data():
    data = []
    for i in range(1, 6):
        data.append(LoadBatch(f'data_batch_{i}'))

    # Stack data
    train_images = np.vstack(batch.get(b'data') for batch in data)
    train_labels = np.hstack(batch.get(b'labels') for batch in data)

    # Center data
    train_images -= np.mean(train_images, dtype='uint8')

    # Test
    test = LoadBatch(f'test_batch')
    test_images = test.get(b'data')
    test_labels = test.get(b'labels')

    # Reshape data
    train_images = train_images.reshape(len(train_images), 3, 32, 32).transpose(0, 2, 3, 1)
    test_images = test_images.reshape(len(test_images), 3, 32, 32).transpose(0, 2, 3, 1)

    # Resize data
    train_images = tf.image.resize_images(train_images, (256, 256))
    test_images = tf.image.resize_images(test_images, (256, 256))

    return (train_images, train_labels), (test_images, test_labels)