from pytorch.structure.sections.blocks.classification import ClassificationBlock


class ClassificationSection:
    def __init__(self, architecture):
        self.classification_block = ClassificationBlock(architecture)

    def build(self):
        return [self.classification_block.build()]
