from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer
from flogo.discovery.regularization.early_stopping import EarlyStopping
from flogo.discovery.regularization.monitors.accuracy_monitor import AccuracyMonitor


class TrainingTask:
    def __init__(self, trainer, epochs: int, architecture, training_dataset, validation_dataset, loss_function: Loss,
                 optimizer: Optimizer, early_stopping=EarlyStopping(AccuracyMonitor(5, 0.005))):
        self.trainer = trainer
        self.epochs = epochs
        self.architecture = architecture
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.early_stopping = early_stopping

    def execute(self):
        return self.trainer(self.epochs,
                            self.architecture,
                            self.training_dataset,
                            self.validation_dataset,
                            self.loss_function,
                            self.optimizer,
                            self.early_stopping
                            ).train()
