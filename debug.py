import os

from data import CTDataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import create_pred
from unet import UNet
import torchvision.transforms as transforms


if __name__ == '__main__':
    runs = [dir for dir in os.listdir('logs/') if os.path.isdir(f'logs/{dir}')]

    # Loader for labeled data
    dataset = CTDataset('train')

    # Count mean number of instances per segmentation on labeled set & non-null pixels & mean number of pixels per instance
    num_objects = []
    non_null_pixels = []
    pixels_per_instance = []
    count_pixels_per_class = []

    for i in range(len(dataset)):
        seg = dataset[i]['seg']
        num_objects.append(len(torch.unique(seg)) - 1)
        non_null_pixels.append(torch.count_nonzero(seg))
        count_pixels_per_class.extend(torch.unique(seg, return_counts=True)[1][1:].tolist())

        # Get number of instance and pixels per instance for each segmentation
        classes, counts = torch.unique(seg, return_counts=True)
        classes, counts = classes.numpy().astype(int), counts.numpy()
        dict_sample = dict(zip(classes, counts))
        dict_sample.pop(0)  # Delete background

        for key in dict_sample:
            pixels_per_instance.append(dict_sample[key])

        _, instances = torch.unique(seg, return_inverse=True)

    # hist = np.unique(num_objects, return_counts=True)
    # df = pd.DataFrame.from_dict(dict(zip(hist[0], hist[1])), orient='index')
    # df.plot(kind='bar',
    #         legend=False,
    #         xlabel='Number of segments',
    #         ylabel='Number of images',
    #         yticks=range(18))
    # plt.show()

    print(count_pixels_per_class)
    hist = np.unique(count_pixels_per_class)
    plt.hist(hist, bins=100)
    plt.xlim(0, 60000)
    plt.xlabel('Number of pixels')
    plt.ylabel('Number of segments')
    plt.show()


    # print(f'Mean number of objects: {np.mean(num_objects)}\n'
    #       f'Std number of objects: {np.std(num_objects)}\n'
    #       f'Max number of objects: {np.max(num_objects)}\n'
    #       f'Median number of objects: {np.median(num_objects)}\n'
    #       f'Min number of objects: {np.min(num_objects)}\n\n'
    #
    #       f'Mean number of non null pixels: {np.mean(non_null_pixels)}\n'
    #       f'Std number of non null pixels: {np.std(non_null_pixels)}\n'
    #       f'Max number of non null pixels: {np.max(non_null_pixels)}\n'
    #       f'Min number of non null pixels: {np.min(non_null_pixels)}\n\n'
    #
    #       f'Mean number of pixels per instance: {np.mean(pixels_per_instance)}\n'
    #       f'Std number of pixels per instance: {np.std(pixels_per_instance)}\n'
    #       f'Max number of pixels per instance: {np.max(pixels_per_instance)}\n'
    #       f'Min number of pixels per instance: {np.min(pixels_per_instance)}')


