from framework.structure.sections.link.classificationsection import ClassificationSection as Classification
from pytorch.structure.blocks.classification import ClassificationBlock


class ClassificationSection:
    def __init__(self, section: Classification):
        self.classification_block = ClassificationBlock(section.section)

    def build(self):
        return [self.classification_block.build()]
