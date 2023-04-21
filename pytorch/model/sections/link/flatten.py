from model.flogo.blocks.flatten import FlogoFlattenBlock
from pytorch.model.layers.flatten import FlattenFunction


class FlattenSection:
    def __init__(self, architecture):
        self.flatten_block = FlattenBlock(architecture)

    def build(self):
        return [self.flatten_block.build()]


class FlattenBlock:
    def __init__(self, block: FlogoFlattenBlock):
        self.flatten = block.flatten

    def build(self):
        return FlattenFunction(self.flatten).build()
