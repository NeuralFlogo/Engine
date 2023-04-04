from compiled.model.layers.convolutional import Conv
from compiled.model.layers.activation import Activation
from compiled.model.layers.pool import Pool


class CompiledInputBlock:
    def __init__(self, conv: Conv, pool: Pool):
        self.conv = conv
        self.pool = pool


class BodyBlock:
    def __init__(self, conv1: Conv, activation: Activation, conv2: Conv):
        self.conv1 = conv1
        self.activation = activation
        self.conv2 = conv2


class OutputBlock:
    def __init__(self, pool: Pool):
        self.pool = pool
