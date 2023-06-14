from framework.structure.metadata import Metadata


class Runnable:
    def __init__(self, structure, metadata: Metadata):
        self.structure = structure
        self.metadata = metadata
