from framework.structure.runnable import Runnable


class StructureLauncher:
    def __init__(self, definition, interpreter):
        self.interpreter = interpreter
        self.definition = definition

    def launch(self) -> Runnable:
        structure, metadata = self.interpreter.generate(self.definition)
        return Runnable(structure, metadata)
