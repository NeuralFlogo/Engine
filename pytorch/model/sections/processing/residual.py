from pytorch.model.layers.activation import ActivationFunction
from pytorch.model.layers.convolution import Conv2d
from pytorch.model.layers.pool import Pool
from pytorch.vocabulary import Kernel, Channel, Stride, Padding, Pooling, Activation, Block


class ResidualSection:
    def __init__(self, architecture):
        body = [BodyBlock(block) for block in architecture[1:-1]]
        self.architecture = [InputBlock(architecture[0])] + body + [OutputBlock(architecture[-1])]

    def build(self):
        return [block.build() for block in self.architecture]


class InputBlock:
    def __init__(self, block):
        self.conv = Conv2d(block[Kernel.Convolutional], block[Channel.In], block[Channel.Out],
                           block[Stride.Convolutional], block[Padding.Convolutional])
        self.pool = Pool(block[Kernel.Pool], block[Stride.Pool],
                         block[Padding.Pool], block[Pooling.Type])

    def build(self):
        return self.conv.build(), self.pool.build()


class BodyBlock:
    def __init__(self, block):
        self.stages = [ResidualBlock(block) for _ in range(block[Block.HiddenSize])]

    def build(self):
        return [res_block.build() for res_block in self.stages]


class ResidualBlock:
    def __init__(self, block):
        self.conv1 = Conv2d(block[Kernel.Convolutional], block[Channel.In],
                            block[Channel.Out], block[Stride.Convolutional], block[Padding.Convolutional])
        self.activation = ActivationFunction(block[Activation.name])
        self.conv2 = Conv2d(block[Kernel.Convolutional], block[Channel.In],
                            block[Channel.Out], block[Stride.Convolutional], block[Padding.Convolutional])

    def build(self):
        return self.conv1.build(), self.activation.build(), self.conv2.build()


class OutputBlock:
    def __init__(self, block):
        self.pool = Pool(block[Kernel.Pool], block[Stride.Pool],
                         block[Padding.Pool], block[Pooling.Type])

    def build(self):
        return self.pool.build()
