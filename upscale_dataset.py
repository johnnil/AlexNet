import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.ndimage import zoom

from tqdm import tqdm

def LoadBatch(filename):
	""" Copied from the dataset website """
	with open('Dataset/'+filename, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def WriteBatch(filename, batch):
    with open('Dataset/'+filename, 'xb') as fo:
        pickle.dump(batch, fo, protocol=pickle.HIGHEST_PROTOCOL)

def upscale_batch(batch, pixels):
    batch = LoadBatch(batch)
    data = batch.get(b'data')
    labels = batch.get(b'labels')

    # Reshape
    data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)

    # Resize
    new_data = []
    for i in tqdm(range(len(data))):
        img = resize(data[i], (224, 224))
        new_data.append(img)

    # Test
    test = new_data[0]
    plt.imshow(test)
    plt.show()

    # Reshape back to original (possibly not necessary to reshape back and forth)
    # Makes it possible to test the images
    new_data = np.array(new_data).transpose(0, 3, 1, 2).flatten()
    
    # Prepare for the home journey
    upscaled_batch = dict()
    upscaled_batch['data'] = new_data
    upscaled_batch['labels'] = labels

    return upscaled_batch

if __name__ == "__main__":
    batch_names = [f'data_batch_{x}' for x in range(1, 6)]
    batch_names.append('test_batch')

    for batch in batch_names:
        print('Upscaling', batch)
        upscaled_batch = upscale_batch(batch, 224)
        print(f'Writing upscaled {batch} to file')
        WriteBatch('upscaled_'+batch, upscaled_batch)
