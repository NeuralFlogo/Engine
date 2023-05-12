from torch import nn


class PytorchLoss:
    def __init__(self, name: str):
        self.function = self.__build(name)

    def __build(self, name):
        return getattr(nn, name)()

    def compute(self, predictions, labels):
        loss = self.function(predictions, labels)
        loss.backward()
        return loss.item()
