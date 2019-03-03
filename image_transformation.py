import numpy as np
import random
from torchvision.transforms import functional as F

class RandomRotation(object):
    """
    Degree(min ,max, dtype: float or int):
        if input one number, degree = (-degree, degree)
    parameters in this function follows the function in torchvision.transforms.RandomRotation
    """
    def __init__(self, degrees):
        self.degrees = degrees


    def __call__(self, data):
        hr, lr = data
        angle = self.get_params(self, degrees)
        # tor