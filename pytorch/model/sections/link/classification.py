from model.flogo.blocks.classification import FlogoClassificationBlock
from pytorch.model.layers.classification import ClassificationFunction


class ClassificationSection:
    def __init__(self, architecture):
        self.classification_block = ClassificationBlock(architecture)

    def build(self):
        return [self.classification_block.build()]


class ClassificationBlock:
    def __init__(self, block: FlogoClassificationBlock):
        self.function = ClassificationFunction(block.classification)

    def build(self):
        return self.function.build()
