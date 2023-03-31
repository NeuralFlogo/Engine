from pytorch.model.layers.activation import ActivationFunction
from pytorch.model.layers.linear import Linear
from pytorch.vocabulary import Channel, Activation


class FeedForward:
    def __init__(self, architecture):
        self.architecture = [LinearBlock(block) for block in architecture]

    def build(self):
        result = []
        for block in self.architecture: result.extend(block.build())
        return result


class LinearBlock:
    def __init__(self, block):
        self.linear = Linear(input_dimension=block[Channel.In], output_dimension=block[Channel.Out])
        self.activation = ActivationFunction(block[Activation.name])

    def build(self):
        return self.linear.build(), self.activation.build()
