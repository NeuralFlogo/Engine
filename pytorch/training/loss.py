from torch import nn

from flogo.training.loss import FlogoLossFunction


class LossFunction:
    def __init__(self, loss: FlogoLossFunction):
        self.name = loss.name

    def build(self):
        return getattr(nn, self.name)()
