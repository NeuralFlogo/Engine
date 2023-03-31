from pytorch.model.layers.flatten import FlattenFunction
from pytorch.vocabulary import Dimension


class Flatten:
    def __init__(self, architecture):
        self.flatten_block = FlattenBlock(architecture)

    def build(self):
        return self.flatten_block.build()


class FlattenBlock:
    def __init__(self, block):
        self.start_dim = block[Dimension.Start]
        self.end_dim = block[Dimension.End]

    def build(self):
        return FlattenFunction(start_dim=self.start_dim, end_dim=self.end_dim).build()
