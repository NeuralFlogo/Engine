class StructureFactory:
    def __init__(self, structure, generator):
        self.generator = generator
        self.structure = structure

    def create_structure(self):
        return self.generator.generate(self.structure)
