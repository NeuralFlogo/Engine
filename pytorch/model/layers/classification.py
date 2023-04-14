import torch


class ClassificationFunction:
    def __init__(self, name, dimension):
        self.name = name
        self.dimension = dimension

    def build(self):
        return getattr(torch.nn, self.name)(dim=self.dimension)
