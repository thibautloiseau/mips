from torch.utils.data import Dataset
import os
import torch
import cv2


class CTDataset(Dataset):
    def __init__(self, mode):
        super().__init__()

        self.samples = []

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

    def __getitem__(self, item):
        slice, seg = self.samples[item]
        data_path = self.samples[item][0]

        slice = torch.from_numpy(cv2.imread(slice, cv2.IMREAD_GRAYSCALE))[None, :, :]
        seg = torch.from_numpy(cv2.imread(seg, cv2.IMREAD_GRAYSCALE))[None, :, :]

        return {'slice': slice, 'seg': seg, 'data_path': data_path}

    def __len__(self):
        return len(self.samples)