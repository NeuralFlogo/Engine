from torch import nn


class FlattenFunction:
    def __init__(self, start_dim, end_dim):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def build(self):
        return nn.Flatten(start_dim=self.start_dim, end_dim=self.end_dim)
