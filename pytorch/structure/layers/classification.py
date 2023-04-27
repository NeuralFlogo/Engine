import torch

from flogo.structure.layers.classification import Classification


class PClassification:
    def __init__(self, classification: Classification):
        self.name = classification.name
        self.dimension = classification.dimension

    def build(self):
        return getattr(torch.nn, self.name)(dim=self.dimension)
