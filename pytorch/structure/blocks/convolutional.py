from framework.structure.blocks import convolutional
from framework.structure.layers import normalization
from framework.structure.layers.activation import Activation
from framework.structure.layers.convolutional import Convolutional
from framework.structure.layers.pool import Pool as PoolComp
from pytorch.structure.layers.activation import PActivation
from pytorch.structure.layers.convolution import PConvolutional
from pytorch.structure.layers.normalization import PNormalization
from pytorch.structure.layers.pool import PPool


class ConvolutionalBlock:
    def __init__(self, block: convolutional.ConvolutionalBlock):
        self.content = []
        for layer in block.content:
            if type(layer) == Convolutional: self.content.append(PConvolutional(layer))
            if type(layer) == Activation: self.content.append(PActivation(layer))
            if type(layer) == PoolComp: self.content.append(PPool(layer))
            if type(layer) == normalization.Normalization: self.content.append(PNormalization(layer))

    def build(self):
        return [layer.build() for layer in self.content]
