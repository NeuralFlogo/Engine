from compiled.model.layers.classification import Classification


class CompiledClassificationBlock:
    def __init__(self, classification: Classification):
        self.classification = classification
