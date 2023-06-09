from torch.utils.data import Dataset
import os
import torch
import cv2
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import numpy as np
from utils import FixRandomSeed


class CTDatasetLabeled(Dataset):
    def __init__(self, mode, transform=None):
        super().__init__()

        self.samples = []
        self.transform = transform
        self.mode = mode

        # Training mode only on initially labeled data
        labeled_idx = [i for i in range(200)]
        np.random.shuffle(labeled_idx)
        idx_split = int((1 - 0.2) * 200)

        if mode == 'train':
            labeled_idx = labeled_idx[:idx_split]
        else:
            labeled_idx = labeled_idx[idx_split:]

        for i in labeled_idx:
            self.samples.append((os.path.join('data', f'X_train', f'{i}.png'),
                                 os.path.join('data', f'y_train', f'{i}.png')))

    def __getitem__(self, item):
        slice, seg = self.samples[item]
        data_path = self.samples[item][0]

        slice = torch.from_numpy(cv2.imread(slice, cv2.IMREAD_GRAYSCALE))[None, :, :].type(torch.FloatTensor)
        seg = torch.from_numpy(cv2.imread(seg, cv2.IMREAD_GRAYSCALE))[None, :, :].type(torch.FloatTensor)

        seed = random.randint(0, 2**32-1)

        # We transform the slice only, not the segmentations
        if self.transform is not None:
            slice, seg = self.transform(slice, seg, seed)

        return {'slice': slice, 'seg': seg, 'data_path': data_path}

    def __len__(self):
        return len(self.samples)


class CTDatasetUnlabeled(Dataset):
    def __init__(self, mode, transform=None):
        super().__init__()

        self.samples = []
        self.transform = transform
        self.mode = mode

        # Training mode only on initially labeled data
        if mode == 'train':
            patient_ids = [i for i in range(200)]

            for i in patient_ids:
                self.samples.append((os.path.join('data', f'X_train', f'{i}.png'),
                                     os.path.join('data', f'y_train', f'{i}.png')))

        # Unlabeled mode to create dataset to create predictions for co-training
        elif mode == 'unlabeled':
            patient_ids = [i for i in range(200, 1000)]

            for i in patient_ids:
                self.samples.append((os.path.join('data', f'X_train', f'{i}.png'),
                                     os.path.join('data', f'y_train', f'{i}.png')))

        # Full mode after creating labels for initially unlabeled data
        elif mode == 'full':
            patient_ids = [i for i in range(1000)]

            for i in patient_ids:
                self.samples.append((os.path.join('data', f'X_train', f'{i}.png'),
                                     os.path.join('data', f'y_train', f'{i}.png')))

        elif mode == 'test':
            patient_ids = [i for i in range(500)]

            for i in patient_ids:
                self.samples.append((os.path.join('data', f'X_test', f'{i}.png'),
                                     ''))

    def __getitem__(self, item):
        slice, seg = self.samples[item]
        data_path = self.samples[item][0]

        slice = torch.from_numpy(cv2.imread(slice, cv2.IMREAD_GRAYSCALE))[None, :, :].type(torch.FloatTensor)

        # Handle test
        if self.mode != 'test':
            seg = torch.from_numpy(cv2.imread(seg, cv2.IMREAD_GRAYSCALE))[None, :, :].type(torch.FloatTensor)
        else:
            seg = ''

        seed = random.randint(0, 2**32-1)

        # We transform the slice only, not the segmentations
        if self.transform is not None:
            slice, seg = self.transform(slice, seg, seed)

        return {'slice': slice, 'seg': seg, 'data_path': data_path}

    def __len__(self):
        return len(self.samples)
