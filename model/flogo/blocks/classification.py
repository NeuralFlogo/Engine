from model.flogo.layers.classification import Classification


class FlogoClassificationBlock:
    def __init__(self, classification: Classification):
        self.classification = classification
