from compiled.model.blocks.flatten import CompiledFlattenBlock
from pytorch.model.layers.flatten import FlattenFunction


class Flatten:
    def __init__(self, architecture):
        self.flatten_block = FlattenBlock(architecture)

    def build(self):
        return self.flatten_block.build()


class FlattenBlock:
    def __init__(self, block: CompiledFlattenBlock):
        self.start_dim = block.flatten.start_dim
        self.end_dim = block.flatten.end_dim

    def build(self):
        return FlattenFunction(start_dim=self.start_dim, end_dim=self.end_dim).build()
