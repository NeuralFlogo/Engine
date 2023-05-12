from torch import nn


class LossMeasurer:
    def __init__(self, name: str):
        self.function = self.__build(name)

    def __build(self, name):
        return getattr(nn, name)()

    def measure(self, predictions, labels):
        return self.function(predictions, labels).item()
