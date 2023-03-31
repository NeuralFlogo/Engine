from vocabulary import Kernel, Channel, Stride, Padding, Activation, Pooling
from model.layers import ActivationFunction, Pool, Conv2d


class ConvolutionalArchitecture:
    def __init__(self, architecture):
        self.architecture = [ConvolutionalBlock(block) for block in architecture]

    def pytorch(self):
        result = []
        for block in self.architecture: result.extend(block.pytorch())
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

    def pytorch(self):
        return self.convolutional_layer.pytorch(), self.activation_function.pytorch(), self.pool_layer.pytorch()
