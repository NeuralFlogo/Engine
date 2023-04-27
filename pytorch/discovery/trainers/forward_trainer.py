import torch

from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer


class ForwardTrainer:
    def __init__(self, epochs: int, model, training_dataset, validation_dataset, loss_function: Loss, optimizer: Optimizer):
        self.epochs = epochs
        self.model = model
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.loss_function = loss_function
        self.optimizer = optimizer

    def train(self):
        for epoch in range(self.epochs):
            self.model.train(True)
            loss = self.__train_model()
            self.model.train(False)
            vloss = self.__validate_model()
            self.__log_epoch_losses(epoch, loss, vloss)

    def __train_model(self):
        running_loss = 0.
        for i, data in enumerate(self.training_dataset, start=1):
            inputs, labels = data
            preds = self.__evaluate(inputs)
            loss = self.loss_function.compute(preds, labels)
            loss.backward()
            running_loss += loss.item()
            self.optimizer.step()
        return self.__epoch_average_loss(running_loss)

    def __validate_model(self):
        running_loss = 0.
        for i, data in enumerate(self.validation_dataset, start=1):
            inputs, labels = data
            preds = self.__evaluate(inputs)
            running_loss += self.loss_function.compute(preds, labels).item()
        return self.__epoch_average_loss(running_loss)

    def __evaluate(self, inputs):
        return self.model(inputs)

    def __log_epoch_losses(self, epoch, train_loss, val_loss):
        print('Training - Validation Loss: Training = {}, Validation = {}\n'.format(train_loss, val_loss, epoch + 1))

    def __epoch_average_loss(self, loss):
        return loss / self.epochs

    def compute_accuracy(self, preds, labels):
        return torch.sum(torch.eq(torch.argmax(preds, dim=1), torch.argmax(labels, dim=1))).item()

    def to_percentage(self, value, batch):
        return 100 * value / len(batch)
