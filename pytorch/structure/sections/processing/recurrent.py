from pytorch.structure.blocks.recurrent import RecurrentBlock


class RecurrentSection:
    def __init__(self, section):
        self.architecture = []
        for block in section.section:
            self.architecture += [RecurrentBlock(block)]

    def build(self):
        return [block.build() for block in self.architecture]

