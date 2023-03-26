from model.vocabulary import Activation
from model.architecture.layers import ClassificationFunction


class ClassificationBlock:
    def __init__(self, architecture):
        self.function = ClassificationFunction(architecture[Activation.name], architecture[Activation.dimension])

    def pytorch(self):
        return self.function.pytorch()


class Classification:
    def __init__(self, architecture):
        self.classification_function = ClassificationBlock(architecture)

    def pytorch(self):
        return self.classification_function.pytorch()
