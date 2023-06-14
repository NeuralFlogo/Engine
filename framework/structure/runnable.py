from framework.architecture.module import Module
from framework.structure.metadata import Metadata


class Runnable:
    def __init__(self, structure, metadata: Metadata):
        self.structure = structure
        self.metadata = metadata

    def get_section(self, index):
        return self.structure[self.metadata.get_start_index(index):self.metadata.get_end_index(index)]

    def get_module(self, index):
        return Module(self.get_section(index), self.metadata.get_section_length(index))

    def as_modules(self):
        return [self.get_module(index) for index in range(len(self.metadata.sections_length))]
