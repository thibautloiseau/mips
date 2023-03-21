import random
import torch
import torchvision.transforms as T

from utils import FixRandomSeed


class TransformLabeled:
    def __init__(self, n_geo=1, n_color=1, magnitude=0.2, mode='train'):
        self.n_geo = n_geo
        self.n_color = n_color
        self.magnitude = magnitude
        self.mode = mode

        self.geo_list = [
            T.RandomRotation(180),
            T.RandomResizedCrop(512, scale=(self.magnitude, 1+self.magnitude)),
            T.RandomAffine(0,
                           translate=(self.magnitude, self.magnitude),
                           scale=(self.magnitude, self.magnitude),
                           shear=(self.magnitude, self.magnitude))
        ]

        self.color_list = [
            T.RandomAdjustSharpness(self.magnitude)
        ]

    def __call__(self, img, label, seed):
        img = T.Normalize(0, 1)(img)

        if self.mode == 'train':
            geo_transform = T.Compose([
                    random.choice(self.geo_list) for _ in range(self.n_geo)
                ])

            with FixRandomSeed(seed):
                img = geo_transform(img)

            with FixRandomSeed(seed):
                label = geo_transform(label)

        return img, label
