from pytorch.structure.blocks.flatten import FlattenBlock


class FlattenSection:
    def __init__(self, section):
        self.flatten_block = FlattenBlock(section.section)

    def build(self):
        return [self.flatten_block.build()]
