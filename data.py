from torch.utils.data import Dataset
import os
import torch
import cv2


class CTDataset(Dataset):
    def __init__(self, mode):
        super().__init__()

        self.samples = []

        if mode == 'train':
            files = sorted(os.listdir(f'data/X_{mode}'))
            patient_ids = list(set([i.split('.')[0] for i in files]))

            for i in patient_ids:
                self.samples.append((os.path.join('data', f'X_{mode}', f'{i}.png'),
                                     os.path.join('data', f'y_{mode}', f'{i}.png')))

        elif mode == 'test':
            files = sorted(os.listdir('data/X_test'))
            patient_ids = list(set([i.split('.')[0] for i in files]))

            for i in patient_ids:
                self.samples.append((os.path.join(f'data', 'X_test', f'{i}.png'),
                                     ''))

    def __getitem__(self, item):
        slice, seg = self.samples[item]

        slice = torch.from_numpy(cv2.imread(slice, cv2.IMREAD_GRAYSCALE))[None, :, :]
        seg = '' if seg == '' else torch.from_numpy(cv2.imread(seg, cv2.IMREAD_GRAYSCALE))[None, :, :]

        return {'slice': slice, 'seg': seg}

    def __len__(self):
        return len(self.samples)
