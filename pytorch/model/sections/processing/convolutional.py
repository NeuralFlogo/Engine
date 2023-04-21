from model.flogo.blocks.convolutional import FlogoConvolutionalBlock
from model.flogo.layers.activation import Activation
from model.flogo.layers.convolutional import Conv
from model.flogo.layers import normalization
from model.flogo.layers.pool import Pool as PoolComp
from pytorch.model.layers.activation import ActivationFunction
from pytorch.model.layers.convolution import Conv2d
from pytorch.model.layers.normalization import Normalization
from pytorch.model.layers.pool import Pool


class ConvolutionalSection:
    def __init__(self, section):
        self.section = [ConvolutionalBlock(block) for block in section]

    def build(self):
        result = []
        for block in self.section: result.extend(block.build())
        return result


class ConvolutionalBlock:
    def __init__(self, block: FlogoConvolutionalBlock):
        self.content = []
        for layer in block.content:
            if type(layer) == Conv: self.content.append(Conv2d(layer))
            if type(layer) == Activation: self.content.append(ActivationFunction(layer))
            if type(layer) == PoolComp: self.content.append(Pool(layer))
            if type(layer) == normalization.Normalization: self.content.append(Normalization(layer))

    def build(self):
        return [layer.build() for layer in self.content]
