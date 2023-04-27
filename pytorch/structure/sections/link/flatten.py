from flogo.structure.blocks import flatten
from pytorch.structure.layers.flatten import PFlatten


class FlattenSection:
    def __init__(self, architecture):
        self.flatten_block = FlattenBlock(architecture)

    def build(self):
        return [self.flatten_block.build()]


class FlattenBlock:
    def __init__(self, block: flatten.FlattenBlock):
        self.flatten = block.flatten

    def build(self):
        return PFlatten(self.flatten).build()
