from framework.structure.blocks import residual
from pytorch.structure.blocks.residual import _ResidualBlock


class ResidualSection:
    def __init__(self, section):
        self.section = [Stage(stage) for stage in section.section]

    def build(self):
        result = []
        for stage in self.section:
            result += stage.build()
        return result


class Stage:
    def __init__(self, block: residual.ResidualBlock):
        self.content = [_ResidualBlock(block) for _ in range(block.hidden_size)]

    def build(self):
        return [block.build() for block in self.content]

