from framework.discovery.hyperparameters.loss import Loss
from framework.discovery.hyperparameters.optimizer import Optimizer


class PytorchTrainer:
    def __init__(self, optimizer: Optimizer, loss_function: Loss):
        self.optimizer = optimizer
        self.loss_function = loss_function

    def train(self, model, training_dataset):
        loss = 0.
        model.train(True)
        for entry in training_dataset:
            inputs, expected = entry.get_input(), entry.get_output()
            loss += self.loss_function.compute(self.__predict(model, inputs), expected)
            self.optimizer.step()
        model.train(False)

    def __predict(self, model, inputs):
        return model(inputs)
