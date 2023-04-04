from compiled.model.layers.linear import Linear
from compiled.model.layers.activation import Activation


class CompiledLinearBlock:
    def __init__(self, linear: Linear, activation: Activation):
        self.linear = linear
        self.activation = activation
