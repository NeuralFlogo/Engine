from framework.structure.structure import Structure


class StructureFactory:
    def __init__(self, structure, generator):
        self.generator = generator
        self.structure = structure

    def create_structure(self):
        structure, metadata = self.generator.generate(self.structure)
        return Structure(structure, metadata)
