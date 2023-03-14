from data import CTDataset
import torch
import numpy as np


if __name__ == '__main__':
    # Loader for labeled data
    dataset = CTDataset('train')

    # Count mean number of instances per segmentation on labeled set & non-null pixels
    num_objects = []
    non_null_pixels = []

    for i in range(len(dataset)):
        seg = dataset[i]['seg']
        num_objects.append(len(torch.unique(seg)) - 1)
        non_null_pixels.append(torch.count_nonzero(seg))

    print(f'Mean number of objects: {np.mean(num_objects)}\n'
          f'Std number of objects: {np.std(num_objects)}\n'
          f'Max number of objects: {np.max(num_objects)}\n\n'
          
          f'Mean number of non null pixels: {np.mean(non_null_pixels)}\n'
          f'Std number of non null pixels: {np.std(non_null_pixels)}\n'
          f'Max number of non null pixels: {np.max(non_null_pixels)}')
