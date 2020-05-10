import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import resize

def LoadBatch(filename):
	""" Copied from the dataset website """
	with open('Dataset/'+filename, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def load_data(upscaled=False):
    upscaled = 'upscaled_' if upscaled else ''
    size = 224 if upscaled else 32

    data = []
    for i in range(1, 6):
        data.append(LoadBatch(f'{upscaled}data_batch_{i}'))

    # Stack data
    train_images = np.vstack(batch.get(b'data') for batch in data)
    train_labels = np.hstack(batch.get(b'labels') for batch in data)

    # Load test data
    test = LoadBatch(f'{upscaled}test_batch')
    test_images = test.get(b'data')
    test_labels = test.get(b'labels')

    # Center data
    train_images = train_images / 255
    test_images = test_images / 255

    # Reshape data
    train_images = train_images.reshape(len(train_images), 3, size, size).transpose(0, 2, 3, 1)
    test_images = test_images.reshape(len(test_images), 3, size, size).transpose(0, 2, 3, 1)

    return (train_images, train_labels), (test_images, test_labels)

load_data()