from torch import nn

from framework.structure.layers.activation import Activation


class PActivation:
    def __init__(self, activation: Activation):
        self.name = activation.name

    def build(self):
        return getattr(nn, self.name)()
