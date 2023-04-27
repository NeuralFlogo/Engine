from flogo.structure.blocks import classification
from pytorch.structure.layers.classification import PClassification


class ClassificationSection:
    def __init__(self, architecture):
        self.classification_block = ClassificationBlock(architecture)

    def build(self):
        return [self.classification_block.build()]


class ClassificationBlock:
    def __init__(self, block: classification.ClassificationBlock):
        self.function = PClassification(block.classification)

    def build(self):
        return self.function.build()
