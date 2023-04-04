from compiled.model.blocks.classification import CompiledClassificationBlock
from pytorch.model.layers.classification import ClassificationFunction


class Classification:
    def __init__(self, architecture):
        self.classification_block = ClassificationBlock(architecture)

    def build(self):
        return self.classification_block.build()


class ClassificationBlock:
    def __init__(self, block: CompiledClassificationBlock):
        self.function = ClassificationFunction(block.classification.name, block.classification.dimension)

    def build(self):
        return self.function.build()
