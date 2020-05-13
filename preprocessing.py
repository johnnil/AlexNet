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

def verify_data(images, labels):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        plt.xlabel(class_names[labels[i][0]])
    plt.show()

def flip_data(images):
    # Flips images vertically
    return images[:, :, ::-1, :].copy()

if __name__ == "__main__":
    pass