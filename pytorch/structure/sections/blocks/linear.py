from flogo.structure.blocks import linear
from flogo.structure.layers.activation import Activation
from flogo.structure.layers.linear import Linear
from pytorch.structure import layers

class LinearBlock:
    def __init__(self, block: linear.LinearBlock):
        self.content = []
        for layer in block.content:
            if type(layer) == Activation: self.content.append(layers.activation.Activation(layer))
            if type(layer) == Linear: self.content.append(layers.linear.PLinear(layer))

    def build(self):
        return [layer.build() for layer in self.content]
