import torch.nn as nn
import torch.nn.functional as F

class ZSSRNet(nn.module):
    def __init__(self, input_channels=3, channels=64, kernel_size=3):
        super(ZSSRNet, self).__init__():
        self.conv_start = nn.Conv2d(input_channels, channels, kernel_size, padding=kernel_size // 2, bias=True)
        self.conv_mid = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias = True)
        self.conv_end = nn.Conv2d(channels, input_channels, kernel_size, padding = kernel_size // 2, bias = True)
        self.relu = nn.Relu()

    def forward(self, x):
        x = self.relu(self.conv_start(x))
        for i in range(6):
            x = self.relu(self.conv_mid(x))
        x = self.relu(self.conv_end(x))

        return x