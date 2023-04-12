from model.model.blocks.linear import FlogoLinearBlock
from model.model.layers.activation import Activation
from pytorch.model.layers.activation import ActivationFunction
from pytorch.model.layers.linear import Linear


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
            if type(layer) == Activation: self.content.append(ActivationFunction(layer.name))
            if type(layer) == FlogoLinearBlock: self.content.append(Linear(layer.input_dimension,
                                                                           layer.output_dimension))

    def build(self):
        return [block.build() for block in self.content]
