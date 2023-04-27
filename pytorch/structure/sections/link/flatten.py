from pytorch.structure.blocks.flatten import FlattenBlock


class FlattenSection:
    def __init__(self, architecture):
        self.flatten_block = FlattenBlock(architecture)

    def build(self):
        return [self.flatten_block.build()]
