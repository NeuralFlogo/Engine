from model.model.layers.classification import Classification


class FlogoClassificationBlock:
    def __init__(self, classification: Classification):
        self.classification = classification
