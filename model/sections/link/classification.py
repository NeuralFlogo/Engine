from vocabulary import Activation
from model.layers import ClassificationFunction


class Classification:
    def __init__(self, architecture):
        self.classification_block = ClassificationBlock(architecture)

    def pytorch(self):
        return self.classification_block.pytorch()


class ClassificationBlock:
    def __init__(self, architecture):
        self.function = ClassificationFunction(architecture[Activation.name], architecture[Activation.dimension])

    def pytorch(self):
        return self.function.pytorch()
