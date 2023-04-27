from torch import optim

from flogo.training.optimizer import FlogoOptimizer


class Optimizer:
    def __init__(self, optimizer: FlogoOptimizer):
        self.name = optimizer.name
        self.model_params = optimizer.model_params
        self.lr = optimizer.lr

    def build(self):
        return getattr(optim, self.name)(self.model_params, self.lr)
