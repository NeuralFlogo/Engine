from pathlib import Path
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from model.flogo.training.flogotraining import FlogoTraining
from pytorch.training.loss import LossFunction
from pytorch.training.optimizer import Optimizer


class Training:
    def __init__(self, train: FlogoTraining):
        self.model = train.model
        self.epochs = train.epochs
        self.training_loader = train.training_loader
        self.validation_loader = train.validation_loader
        self.optimizer = Optimizer(train.optimizer).build()
        self.loss_function = LossFunction(train.loss_function).build()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter("runs/model_{}".format(self.timestamp))

    def train(self):
        best_vloss = 1_000_000.
        for epoch in range(self.epochs):
            self.model.train(True)
            avg_loss = self.__train_epoch(epoch)
            self.model.train(False)
            avg_vloss = self.__validate_epoch()
            self.__log_epoch_losses(epoch, avg_loss, avg_vloss)
            self.writer.flush()
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                self.__save_model(epoch)
        self.writer.close()

    def __train_epoch(self, epoch):
        loss = 0.
        for i, data in enumerate(self.training_loader, start=1):
            inputs, labels = data
            self.optimizer.zero_grad()
            loss = self.__compute_loss(inputs, labels)
            loss.backward()
            loss += loss.item()
            self.optimizer.step()
            self.__log_train_loss(self.__training_count(epoch, i), loss)
        return self.__epoch_average_loss(loss)

    def __log_train_loss(self, training_count, loss):
        self.writer.add_scalar('Train Loss', loss, training_count)

    def __validate_epoch(self):
        vloss = 0.
        for data in enumerate(self.validation_loader, start=1):
            inputs, labels = data
            vloss += self.__compute_loss(inputs, labels).item()
        return self.__epoch_average_loss(vloss)

    def __log_epoch_losses(self, epoch, avg_loss, avg_vloss):
        self.writer.add_scalars('Training - Validation Loss',
                                {'Training': avg_loss, 'Validation': avg_vloss},
                                epoch + 1)

    def __save_model(self, epoch):
        Path('/models').mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), 'models/model_{}_{}'.format(self.timestamp, epoch))

    def __compute_loss(self, inputs, labels):
        return self.loss_function(self.__evaluate(inputs), labels)

    def __evaluate(self, inputs):
        return self.model(inputs)

    def __training_count(self, epoch, i):
        return epoch * len(self.training_loader) + i

    def __epoch_average_loss(self, loss):
        return loss / self.epochs
