from pytorch.structure.sections.blocks.recurrent import RecurrentBlock


class RecurrentSection:
    def __init__(self, section):
        self.architecture = []
        for block in section.section:
            self.architecture += (RecurrentBlock(block) for _ in range(block.hidden_size))

    def build(self):
        return [block.build() for block in self.architecture]

