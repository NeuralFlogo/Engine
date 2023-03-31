from pytorch.model.layers.classification import ClassificationFunction
from pytorch.vocabulary import Activation


class Classification:
    def __init__(self, architecture):
        self.classification_block = ClassificationBlock(architecture)

    def build(self):
        return self.classification_block.build()


class ClassificationBlock:
    def __init__(self, architecture):
        self.function = ClassificationFunction(architecture[Activation.name], architecture[Activation.dimension])

    def build(self):
        return self.function.build()
