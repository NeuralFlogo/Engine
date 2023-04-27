from pytorch.structure.sections.blocks.convolutional import ConvolutionalBlock


class ConvolutionalSection:
    def __init__(self, section):
        self.section = [ConvolutionalBlock(block) for block in section.section]

    def build(self):
        result = []
        for block in self.section: result.extend(block.build())
        return result
