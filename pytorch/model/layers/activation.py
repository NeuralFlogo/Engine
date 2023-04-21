from torch import nn

from model.flogo.layers.activation import Activation


class ActivationFunction:
    def __init__(self, activation: Activation):
        self.name = activation.name

    def build(self):
        return getattr(nn, self.name)()
