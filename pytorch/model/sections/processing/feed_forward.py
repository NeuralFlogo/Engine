from compiled.model.blocks.linear import CompiledLinearBlock
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
    def __init__(self, block: CompiledLinearBlock):
        self.linear = Linear(block.linear.input_dimension, block.linear.output_dimension)
        self.activation = ActivationFunction(block.activation.name)

    def build(self):
        return self.linear.build(), self.activation.build()
