from pytorch.model.layers.activation import ActivationFunction
from pytorch.model.layers.convolution import Conv2d
from pytorch.model.layers.pool import Pool
from pytorch.vocabulary import Kernel, Channel, Stride, Padding, Activation, Pooling


class Convolutional:
    def __init__(self, architecture):
        self.architecture = [ConvolutionalBlock(block) for block in architecture]

    def build(self):
        result = []
        for block in self.architecture: result.extend(block.build())
        return result


class ConvolutionalBlock:
    def __init__(self, block):
        self.convolutional_layer = Conv2d(block[Kernel.Convolutional],
                                          block[Channel.In],
                                          block[Channel.Out],
                                          block[Stride.Convolutional],
                                          block[Padding.Convolutional])
        self.activation_function = ActivationFunction(block[Activation.name])
        self.pool_layer = Pool(block[Kernel.Pool],
                               block[Stride.Pool],
                               block[Padding.Pool],
                               block[Pooling.type])

    def build(self):
        return self.convolutional_layer.build(), self.activation_function.build(), self.pool_layer.build()
