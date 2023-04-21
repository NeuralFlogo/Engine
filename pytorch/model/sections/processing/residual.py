import torch.nn

from model.flogo.blocks.residual import FlogoResidualBlock
from pytorch.model.layers.activation import ActivationFunction
from pytorch.model.layers.convolution import Conv2d
from pytorch.model.layers.normalization import Normalization


class ResidualSection:
    def __init__(self, section):
        self.section = [Stage(stage) for stage in section]

    def build(self):
        result = []
        for stage in self.section:
            result += stage.build()
        return result


class Stage:
    def __init__(self, block: FlogoResidualBlock):
        self.content = [_ResidualBlock(block) for _ in range(block.hidden_size)]

    def build(self):
        return [block.build() for block in self.content]


class _ResidualBlock:
    def __init__(self, block: FlogoResidualBlock):
        self.stage1 = Block(Conv2d(block.conv1), Normalization(block.norm1), ActivationFunction(block.activation))
        self.stage2 = Block(Conv2d(block.conv2), Normalization(block.norm2))
        self.downsample = block.downsample
        self.activation = ActivationFunction(block.activation)

    def build(self):
        return ResidualBlock(self)


class Block:
    def __init__(self, *content):
        self.content = content

    def build(self):
        return torch.nn.Sequential(*[layer.build() for layer in self.content])


class ResidualBlock(torch.nn.Module):
    def __init__(self, block: _ResidualBlock):
        super().__init__()
        self.block1 = block.stage1.build()
        self.block2 = block.stage2.build()
        self.downsample = block.downsample
        self.activation = block.activation.build()

    def forward(self, x):
        residual = x
        out = self.block1(x)
        out = self.block2(out)
        if self.downsample:
            residual = self.downsample(x)
        return self.activation(out + residual)
