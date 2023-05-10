from flogo.structure.layers.activation import Activation
from flogo.structure.layers.convolutional import Convolutional
from flogo.structure.layers.normalization import Normalization


class ResidualBlock:
    def __init__(self, in_channels: int, out_channels: int, activation:str, stride=1, downsample=None, hidden_size=1):
        self.conv1 = Convolutional(in_channels, out_channels, stride=stride, padding=1)
        self.norm1 = Normalization(out_channels)
        self.activation = Activation(activation)
        self.conv2 = Convolutional(out_channels, out_channels, stride=1, padding=1)
        self.norm2 = Normalization(out_channels)
        self.downsample = downsample
        self.activation = Activation(activation)
        self.hidden_size = hidden_size

