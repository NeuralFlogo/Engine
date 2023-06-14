from framework.structure.metadata import Metadata


class Structure:
    def __init__(self, structure, metadata: Metadata):
        self.content = structure
        self.metadata = metadata
