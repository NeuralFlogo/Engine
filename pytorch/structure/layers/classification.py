import torch

from framework.structure.blocks.classification import ClassificationBlock


class PClassification:
    def __init__(self, classification: ClassificationBlock):
        self.name = classification.classification.name
        self.dimension = classification.classification.dimension

    def build(self):
        return getattr(torch.nn, self.name)(dim=self.dimension)
