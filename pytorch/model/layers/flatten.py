from torch import nn

from model.flogo.layers.flatten import Flatten


class FlattenFunction:
    def __init__(self, flatten: Flatten):
        self.start_dim = flatten.start_dim
        self.end_dim = flatten.end_dim

    def build(self):
        return nn.Flatten(start_dim=self.start_dim, end_dim=self.end_dim)
