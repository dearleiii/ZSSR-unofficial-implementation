import numpy as np
import random
from torchvision.transforms import functional as F

class RandomRotation(object):
    """
    Degree(min ,max, dtype: float or int):
        if input one number, degree = (-degree, degree)
    parameters in this function follows the function in torchvision.transforms.RandomRotation
    """
    def __init__(self, degrees, resample=False, expand=False, center=None):
        # self.degrees = degrees
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = np.random.choice(degrees)
        return angle

    def __call__(self, data):
        hr, lr = data
        angle = self.get_params(self.degrees)
        return F.rotate(hr, angle, self.resample, self.expand, self.center), \
                F.rotate(lr, angle, self.resample, self.expand, self.center)

class RandomHorizontalFlip(object):
    """
    Random flip the images with p = 0.5
    Depend on:
    torchvision.transforms.functional.hflip(img)
        Parameters:	img (PIL Image) – Image to be flipped.
        Returns:	Horizontall flipped image.
        Return type:	PIL Image
    """
    def __call__(self, data):
        # data: hr, lr
        hr, lr =  data
        if random.random() < 0.5:
            return F.hflip(hr), F.hflip(lr)
        return hr, lr

class RandomVerticalFlip(object):
    def __call__(self, data):
        hr, lr = data
        if random.random() < 0.5:
            return F.vflip(hr), F.vflip(lr)
        return hr, lr

class RandomCrop(object):
    """Crop the given PIL Image at a random location.
        Args:
            size (sequence or int): Desired output size of the crop. If size is an
                int instead of sequence like (h, w), a square crop (size, size) is
                made.
            padding (int or sequence, optional): Optional padding on each border
                of the image. Default is 0, i.e no padding. If a sequence of length
                4 is provided, it is used to pad left, top, right, bottom borders
                respectively.
        """
    """
    Depend on :
        torchvision.transforms.functional.crop(img, i, j, h, w)

    img (PIL Image) – Image to be cropped.
    i – Upper pixel coordinate.
    j – Left pixel coordinate.
    h – Height of the cropped image.
    w – Width of the cropped image.
    
        torchvision.transforms.functional.pad(img, padding, fill=0, padding_mode='constant')
    """
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(data, output_size):
        hr, lr = data
        h, w = hr.size
        th, tw = output_size
        if w == tw or h == th:
            return 0, 0, h, w

        if w < tw or h < th:
            th, tw = h // 2, w // 2

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, data):
        hr, lr =  data
        if self.padding > 0:
            hr = F.pad(hr, self.padding)
            lr = F.pad(lr, self.padding)

        i, j, h, w = self.get_params(data, self.size)
        return F.crop(hr, i, h, h, w), F.crop(lr, i, j, h, w)


class ToTensor(object):
    """
    Depend:
    torchvision.transforms.functional.to_tensor(pic)
        Convert a PIL Image or numpy.ndarray to tensor.
    """
    def __call__(self, data):
        hr, lr = data
        return F.to_tensor(hr), F.to_tensor(lr)