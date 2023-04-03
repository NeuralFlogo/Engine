from pytorch.model.layers.activation import ActivationFunction
from pytorch.model.layers.convolution import Conv2d
from pytorch.model.layers.pool import Pool
from pytorch.vocabulary import Kernel, Channel, Stride, Padding, Activation, Pooling, Layers


class Convolutional:
    def __init__(self, architecture):
        self.architecture = [ConvolutionalBlock(block) for block in architecture]

    def build(self):
        result = []
        for block in self.architecture: result.extend(block.build())
        return [result]


class ConvolutionalBlock:
    def __init__(self, block):
        self.result = []
        for layer in block:
            if layer[Layers.Type] == "Convolutional": self.result.append(Conv2d(layer[Kernel.Convolutional],
                                                                                layer[Channel.In],
                                                                                layer[Channel.Out],
                                                                                layer[Stride.Convolutional],
                                                                                layer[Padding.Convolutional]))
            if layer[Layers.Type] == "Activation": self.result.append(ActivationFunction(layer[Activation.name]))
            if layer[Layers.Type] == "Pool": self.result.append(Pool(layer[Kernel.Pool],
                                                                     layer[Stride.Pool],
                                                                     layer[Padding.Pool],
                                                                     layer[Pooling.Type]))

    def build(self):
        return [block.build() for block in self.result]
