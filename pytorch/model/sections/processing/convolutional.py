from compiled.model.blocks.convolutional import CompiledConvolutionalBlock
from compiled.model.layers.activation import Activation
from compiled.model.layers.convolutional import Conv
from compiled.model.layers.pool import Pool as PoolComp
from pytorch.model.layers.activation import ActivationFunction
from pytorch.model.layers.convolution import Conv2d
from pytorch.model.layers.pool import Pool


class ConvolutionalSection:
    def __init__(self, section):
        self.section = [ConvolutionalBlock(block) for block in section]

    def build(self):
        result = []
        for block in self.section: result.extend(block.build())
        return result


class ConvolutionalBlock:
    def __init__(self, block: CompiledConvolutionalBlock):
        self.content = []
        for layer in block.content:
            if type(layer) == Conv: self.content.append(Conv2d(layer.kernel,
                                                               layer.channel_in,
                                                               layer.channel_out,
                                                               layer.stride,
                                                               layer.padding))
            if type(layer) == Activation: self.content.append(ActivationFunction(layer.name))
            if type(layer) == PoolComp: self.content.append(Pool(layer.kernel,
                                                                 layer.stride,
                                                                 layer.padding,
                                                                 layer.pool_type))

    def build(self):
        return [block.build() for block in self.content]
