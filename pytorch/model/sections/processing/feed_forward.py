from compiled.model.blocks.linear import CompiledLinearBlock
from pytorch.model.layers.activation import ActivationFunction
from pytorch.model.layers.linear import Linear


class FeedForward:
    def __init__(self, architecture):
        self.architecture = [LinearBlock(block) for block in architecture]

    def build(self):
        result = []
        for block in self.architecture: result.extend(block.build())
        return result


class LinearBlock:
    def __init__(self, block: CompiledLinearBlock):
        self.linear = Linear(input_dimension=block.linear.input_dimension, output_dimension=block.linear.output_dimension)
        self.activation = ActivationFunction(block.activation.name)

    def build(self):
        return self.linear.build(), self.activation.build()
