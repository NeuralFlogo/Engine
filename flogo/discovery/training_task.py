from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer


class TrainingTask:
    def __init__(self, epochs: int, architecture, training_dataset, validation_dataset, loss_function: Loss, optimizer: Optimizer, trainer):
        self.epochs = epochs
        self.architecture = architecture
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.trainer = trainer

    def execute(self):
        self.trainer(self.epochs,
                     self.architecture,
                     self.training_dataset,
                     self.validation_dataset,
                     self.loss_function,
                     self.optimizer
                     ).train()
