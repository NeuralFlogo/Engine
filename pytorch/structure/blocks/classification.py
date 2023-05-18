from pytorch.structure.layers.classification import PClassification
from framework.structure.blocks import classification


class ClassificationBlock:
    def __init__(self, block: classification.ClassificationBlock):
        self.function = PClassification(block)

    def build(self):
        return self.function.build()
