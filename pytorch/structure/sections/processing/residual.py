import torch.nn

from flogo.structure.blocks import residual
from pytorch.structure.layers.activation import PActivation
from pytorch.structure.layers.convolution import PConvolutional
from pytorch.structure.layers.normalization import PNormalization


class ResidualSection:
    def __init__(self, section):
        self.section = [Stage(stage) for stage in section]

    def build(self):
        result = []
        for stage in self.section:
            result += stage.build()
        return result


class Stage:
    def __init__(self, block: residual.ResidualBlock):
        self.content = [_ResidualBlock(block) for _ in range(block.hidden_size)]

    def build(self):
        return [block.build() for block in self.content]


class _ResidualBlock:
    def __init__(self, block: residual.ResidualBlock):
        self.stack1 = Stack(PConvolutional(block.conv1), PNormalization(block.norm1), PActivation(block.activation))
        self.stack2 = Stack(PConvolutional(block.conv2), PNormalization(block.norm2))
        self.downsample = block.downsample
        self.activation = PActivation(block.activation)

    def build(self):
        return ResidualBlock(self)


class Stack:
    def __init__(self, *content):
        self.content = content

    def build(self):
        return torch.nn.Sequential(*[layer.build() for layer in self.content])


class ResidualBlock(torch.nn.Module):
    def __init__(self, block: _ResidualBlock):
        super().__init__()
        self.block1 = block.stack1.build()
        self.block2 = block.stack2.build()
        self.downsample = block.downsample
        self.activation = block.activation.build()

    def forward(self, x):
        residual = x
        out = self.block1(x)
        out = self.block2(out)
        if self.downsample:
            residual = self.downsample(x)
        return self.activation(out + residual)
