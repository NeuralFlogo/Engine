class StructureFactory:
    def __init__(self, structure, generator):
        self.structure = structure
        self.generator = generator

    def create_structure(self):
        return self.generator.generate(self.structure)
