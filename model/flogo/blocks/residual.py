from model.flogo.layers.convolutional import Conv
from model.flogo.layers.pool import Pool


class FlogoInputBlock:
    def __init__(self, conv: Conv, pool: Pool):
        self.conv = conv
        self.pool = pool


class FlogoBodyBlock:
    def __init__(self, content, hidden_size):
        self.content = content
        self.hidden_size = hidden_size


class FlogoOutputBlock:
    def __init__(self, pool: Pool):
        self.pool = pool
