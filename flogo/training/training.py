from flogo.training.loss import FlogoLossFunction
from flogo.training.optimizer import FlogoOptimizer


class FlogoTraining:
    def __init__(self, epochs: int, model, training_loader, validation_loader, loss_function: FlogoLossFunction, optimizer: FlogoOptimizer):
        self.epochs = epochs
        self.model = model
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
