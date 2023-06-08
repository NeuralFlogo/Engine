from framework.discovery.regularization.early_stopping import EarlyStopping
from framework.discovery.regularization.monitors.growth_monitor import GrowthMonitor


class TrainingTask:
    def __init__(self, trainer, validator, early_stopping: EarlyStopping = EarlyStopping(GrowthMonitor(5, 0.001))):
        self.trainer = trainer
        self.validator = validator
        self.early_stopping = early_stopping

    def execute(self, epochs, model, training_dataset, validation_dataset=None):
        for epoch in range(epochs):
            self.trainer.train(model, training_dataset)
            if validation_dataset:
                if self.__is_model_trained(model, validation_dataset): break
        return model

    def __is_model_trained(self, model, validation_dataset):
        return self.early_stopping.check(self.validator.validate(model, validation_dataset))
