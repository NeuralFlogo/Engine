import torch.nn
from torch import nn


class FeedForward:
    def __init__(self, architecture):
        self.architecture = [LinearBlock(block) for block in architecture]

    def pytorch(self):
        result = []
        for block in self.architecture: result.extend(block.pytorch())
        return result


class LinearBlock:
    def __init__(self, block):
        self.linear = Linear(block["input_dimension"], block["output_dimension"])
        self.activation = Activation(block["activation"])

    def pytorch(self):
        return self.linear.pytorch(), self.activation.pytorch()


class Linear:
    def __init__(self, input_dimension, output_dimension):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

    def pytorch(self):
        return torch.nn.Linear(in_features=self.input_dimension, out_features=self.output_dimension)


class Activation:
    def __init__(self, name):
        self.name = name

    def pytorch(self):
        return getattr(nn, self.name)()
