from torch import nn


class ClassificationFunction:
    def __init__(self, name, dimension):
        self.name = name
        self.dimension = dimension

    def build(self):
        return getattr(nn, self.name)(dim=self.dimension)
