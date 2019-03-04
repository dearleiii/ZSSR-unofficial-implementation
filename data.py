import PIL 
import sys
import random

import torch
from torchvision import transforms
from source_traget_transformations import *

class DataSampler:
    def __init__(self, img, sr_factor, crop_size):
        self.img = img
        self.sr_factor = sr_factor
        self.crop_size = crop_size
        self.pairs = self.create_hr_lr_pairs()

        self.transform = transforms.Compose([
            RandomRotation([0, 90, 180, 270]),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomCrop(crop_size),
            ToTensor(),
        ])

    def create_hr_lr_pairs(self):
        """
        with given img (hr)
        create all pairs possible (hr, lr)
            hr: img_origin with certain downsample factor
            lr: hr down sample with scale factor
        :return: pairs

        Dependence:
            PIL.Image.resize(size, resample=0)
        """
        s_side = min(self.img.size[0: 2])
        l_side = max(self.img.size[0: 2])

        factors = []
        for i in range(s_side // 5, s_side + 1):
            ds_s_side = i
            zoom = float(ds_s_side) / s_side
            ds_l_side = l_side * zoom
            # append the scale factor if (ds_l_side, ds_s_side) can produce lr pair
            if ds_l_side % self.sr_factor == 0 and ds_s_side % self.sr_factor == 0:
                factors.append(zoom)

        pairs = []
        for zoom in factors:
            hr = self.img.resize((int(self.img.size[0] * zoom), int(self.img.size[1]* zoom)), resample=PIL.Image.BICUBIC)
            lr = hr.resize((int(hr.size[0] / self.sr_factor), int(hr.size[1] / self.sr_factor)), resample=PIL.Image.BICUBIC)

            lr = lr.resize(hr.size, resample=PIL.Image.BICUBIC)
            pairs.append((hr, lr))

        return pairs

    def generate_data(self):
        while True:
            # random choose from pairs
            hr, lr =
            hr_tensor, lr_tensor = self.transform(hr, lr)


if __name__ == '__main__':
    img = PIL.Image.open(sys.argv[1])
    sampler = DataSampler(img, 2)
    for x in sampler.generate_data():
        hr, lr = x

