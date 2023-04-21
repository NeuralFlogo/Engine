from model.flogo.blocks.linear import FlogoLinearBlock
from model.flogo.layers.activation import Activation
from model.flogo.layers.linear import Linear
from pytorch.model.layers import linear, activation


class FeedForwardSection:
    def __init__(self, section):
        self.section = [LinearBlock(block) for block in section]

    def build(self):
        result = []
        for block in self.section: result.extend(block.build())
        return result


class LinearBlock:
    def __init__(self, block: FlogoLinearBlock):
        self.content = []
        for layer in block.content:
            if type(layer) == Activation: self.content.append(activation.Activation(layer))
            if type(layer) == Linear: self.content.append(linear.Linear(layer))

    def build(self):
        return [layer.build() for layer in self.content]
