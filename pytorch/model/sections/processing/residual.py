from compiled.model.blocks.residual import CompiledInputBlock, CompiledBodyBlock, CompiledOutputBlock
from compiled.model.layers.activation import Activation
from compiled.model.layers.convolutional import Conv
from compiled.model.layers.linear import Linear
from pytorch.model.layers.activation import ActivationFunction
from pytorch.model.layers.convolution import Conv2d
from pytorch.model.layers.pool import Pool
from compiled.model.layers.pool import Pool as PoolComp


class ResidualSection:
    def __init__(self, section):
        body = [BodyBlock(block) for block in section[1:-1]]
        self.section = [InputBlock(section[0])] + body + [OutputBlock(section[-1])]

    def build(self):
        return [block.build() for block in self.section]


class InputBlock:
    def __init__(self, block: CompiledInputBlock):
        self.conv = Conv2d(block.conv.kernel, block.conv.channel_in,
                           block.conv.channel_out, block.conv.stride, block.conv.padding)
        self.pool = Pool(block.pool.kernel, block.pool.stride, block.pool.padding, block.pool.pool_type)

    def build(self):
        return self.conv.build(), self.pool.build()


class BodyBlock:
    def __init__(self, block: CompiledBodyBlock):
        self.stages = [ResidualBlock(block.content) for _ in range(block.hidden_size)]

    def build(self):
        return [res_block.build() for res_block in self.stages]


class ResidualBlock:
    def __init__(self, block):
        self.content = []
        for layer in block:
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
            if type(layer) == Linear: self.content.append(Linear(layer.input_dimension,
                                                                 layer.output_dimension))

    def build(self):
        return [layer.build() for layer in self.content]


class OutputBlock:
    def __init__(self, block: CompiledOutputBlock):
        self.pool = Pool(block.pool.kernel, block.pool.stride, block.pool.padding, block.pool.pool_type)

    def build(self):
        return self.pool.build()
