import torch

from model.flogo.layers.convolutional import Conv


class Conv2d:
    def __init__(self, convolutional: Conv):
        self.kernel_size = convolutional.kernel
        self.in_channels = convolutional.in_channels
        self.out_channels = convolutional.out_channels
        self.stride = convolutional.stride
        self.padding = convolutional.padding

    def build(self):
        return torch.nn.Conv2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                               in_channels=self.in_channels, out_channels=self.out_channels)
