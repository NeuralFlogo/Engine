from torch import nn


class ActivationFunction:
    def __init__(self, name):
        self.name = name

    def build(self):
        return getattr(nn, self.name)()
