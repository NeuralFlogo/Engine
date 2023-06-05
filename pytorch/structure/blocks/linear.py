from framework.structure.blocks import linear
from framework.structure.layers.activation import Activation
from framework.structure.layers.linear import Linear
from pytorch.structure.layers.linear import PLinear
from pytorch.structure.layers.activation import PActivation


class LinearBlock:
    def __init__(self, block: linear.LinearBlock):
        self.content = []
        for layer in block.content:
            if type(layer) == Activation: self.content.append(PActivation(layer))
            if type(layer) == Linear: self.content.append(PLinear(layer))

    def build(self):
        return [layer.build() for layer in self.content]
