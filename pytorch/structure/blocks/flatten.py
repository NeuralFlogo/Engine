from flogo.structure.blocks import flatten
from pytorch.structure.layers.flatten import PFlatten


class FlattenBlock:
    def __init__(self, block: flatten.FlattenBlock):
        self.flatten = block.flatten


    def build(self):
        return PFlatten(self.flatten).build()
