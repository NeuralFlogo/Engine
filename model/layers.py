import torch
from torch import nn
from torch.nn import MaxPool2d, AvgPool2d


class Linear:
    def __init__(self, input_dimension, output_dimension):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

    def pytorch(self):
        return nn.Linear(in_features=self.input_dimension, out_features=self.output_dimension)


class Activation:
    def __init__(self, name):
        self.name = name

    def pytorch(self):
        return getattr(nn, self.name)()


class Pool:
    def __init__(self, pool, stride, padding, pooling_type):
        self.kernel = pool
        self.stride = stride
        self.padding = padding
        self.pooling_type = pooling_type

    def pytorch(self):
        if self.pooling_type == "Max":
            return MaxPool2d(kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        if self.pooling_type == "Avg":
            return AvgPool2d(kernel_size=self.kernel, stride=self.stride, padding=self.padding)


class ActivationFunction:
    def __init__(self, name):
        self.name = name

    def pytorch(self):
        return getattr(nn, self.name)()


class ClassificationFunction:
    def __init__(self, name, dimension):
        self.name = name
        self.dimension = dimension

    def pytorch(self):
        return getattr(nn, self.name)(dim=self.dimension)


class Conv2d:
    def __init__(self, kernel_size, in_channels, out_channels, stride, padding):
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

    def pytorch(self):
        return torch.nn.Conv2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                               in_channels=self.in_channels, out_channels=self.out_channels)
