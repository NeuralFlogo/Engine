from model.layers import ClassificationFunction


class ClassificationBlock:
    def __init__(self, architecture):
        self.function = ClassificationFunction(architecture["name"], architecture["dimension"])

    def pytorch(self):
        return self.function.pytorch()


class Classification:
    def __init__(self, architecture):
        self.block = ClassificationBlock(architecture)

    def pytorch(self):
        return self.block.pytorch()
