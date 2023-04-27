from pathlib import Path

import torch

from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer


def compute_accuracy(preds, labels):
    # return torch.sum(torch.eq(torch.argmax(preds, dim=1), labels)).item()
    return torch.sum(torch.eq(torch.argmax(preds, dim=1), torch.argmax(labels, dim=1))).item()


def to_percentage(value, batch):
    return 100 * value / len(batch)


class ForwardStrategy:
    def __init__(self, epochs: int, model, training_dataset, validation_dataset,
                 loss_function: Loss, optimizer: Optimizer):
        super().__init__(epochs, model, training_dataset, validation_dataset, loss_function, optimizer)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train(True)
            avg_loss = self.__train_model()
            self.model.train(False)
            avg_vloss = self.__validate_model()
            self.__save_model(epoch)
            self.__log_epoch_losses(epoch, avg_loss, avg_vloss)

    def __train_model(self):
        running_loss = 0.
        for i, data in enumerate(self.training_dataset, start=1):
            inputs, labels = data
            print(inputs.shape)
            self.optimizer.zero_grad()
            preds = self.__evaluate(inputs)
            loss = self.__compute_loss(preds, labels)
            loss.backward()
            running_loss += loss.item()
            self.optimizer.step()
        return self.__epoch_average_loss(running_loss)

    def __validate_model(self):
        vloss = 0.
        for i, data in enumerate(self.validation_dataset, start=1):
            inputs, labels = data
            preds = self.__evaluate(inputs)
            vloss += self.__compute_loss(preds, labels).item()
        return self.__epoch_average_loss(vloss)

    def __evaluate(self, inputs):
        return self.model(inputs)

    def __compute_loss(self, preds, labels):
        return self.loss_function(preds, labels)

    def __log_epoch_losses(self, epoch, train_loss, val_loss):
        self.writer.add_scalars('Training - Validation Loss',
                                {'Training': train_loss, 'Validation': val_loss},
                                epoch + 1)

    def __save_model(self, epoch):
        Path('models').mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), 'models/model_{}_{}'.format(self.model.__class__.name__, epoch))

    def __training_count(self, epoch, i):
        return epoch * len(self.training_dataset) + i

    def __epoch_average_loss(self, loss):
        return loss / self.epochs
