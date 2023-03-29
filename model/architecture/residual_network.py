from model.vocabulary import Kernel, Channel, Stride, Padding, Pooling, Activation, Block
from model.architecture.layers import Conv2d, Pool, ActivationFunction


class ResNet:
    def __init__(self, architecture):
        body = [BodyBlock(block) for block in architecture[1:-1]]
        self.architecture = [InputBlock(architecture[0])] + body + [OutputBlock(architecture[-1])]

    def pytorch(self):
        return [block.pytorch() for block in self.architecture]


class InputBlock:
    def __init__(self, block):
        self.conv = Conv2d(block[Kernel.Convolutional], block[Channel.In], block[Channel.Out],
                           block[Stride.Convolutional], block[Padding.Convolutional])
        self.pool = Pool(block[Kernel.Pool], block[Stride.Pool],
                         block[Padding.Pool], block[Pooling.type])

    def pytorch(self):
        return self.conv.pytorch(), self.pool.pytorch()


class BodyBlock:
    def __init__(self, block):
        self.stages = [ResidualBlock(block) for _ in range(block[Block.HiddenSize])]

    def pytorch(self):
        return [res_block.pytorch() for res_block in self.stages]


class ResidualBlock:
    def __init__(self, block):
        self.conv1 = Conv2d(block[Kernel.Convolutional], block[Channel.In],
                            block[Channel.Out], block[Stride.Convolutional], block[Padding.Convolutional])
        self.activation = ActivationFunction(block[Activation.name])
        self.conv2 = Conv2d(block[Kernel.Convolutional], block[Channel.In],
                            block[Channel.Out], block[Stride.Convolutional], block[Padding.Convolutional])

    def pytorch(self):
        return self.conv1.pytorch(), self.activation.pytorch(), self.conv2.pytorch()


class OutputBlock:
    def __init__(self, block):
        self.pool = Pool(block[Kernel.Pool], block[Stride.Pool],
                         block[Padding.Pool], block[Pooling.type])

    def pytorch(self):
        return self.pool.pytorch()
