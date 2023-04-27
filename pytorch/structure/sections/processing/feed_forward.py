from pytorch.structure.blocks.linear import LinearBlock


class FeedForwardSection:
    def __init__(self, section):
        self.section = [LinearBlock(block) for block in section.section]

    def build(self):
        result = []
        for block in self.section: result.extend(block.build())
        return result
