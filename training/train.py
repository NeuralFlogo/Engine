import torch
from torch.utils.tensorboard import SummaryWriter

from preprocesing.NumericProcessor import one_hot_encode


class Training:
    def __init__(self, model, training_loader, validation_loader, loss_function, optimizer, epochs):
        self.model = model
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.epochs = epochs
        self.writer = SummaryWriter('runs/model')

    def train(self):
        best_vloss = 1_000_000.
        for epoch in range(self.epochs):
            self.model.train(True)
            avg_loss = self.__train_epoch(epoch)
            self.model.train(False)
            running_vloss = 0.0
            avg_vloss = self.__validate_epoch(running_vloss)
            self.writer.add_scalars('Training vs Validation Loss',
                                    {'Training': avg_loss, 'Validation': avg_vloss},
                                    epoch + 1)
            self.writer.flush()
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                torch.save(self.model.state_dict(), 'model_{}'.format(epoch))
        self.writer.close()

    def __train_epoch(self, epoch):
        running_loss = 0.
        last_loss = 0.
        for i, data in enumerate(self.training_loader):
            inputs, labels = data
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000
                tb_x = epoch * len(self.training_loader) + i + 1
                self.writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
        return last_loss

    def __validate_epoch(self, running_loss):
        for i, data in enumerate(self.validation_loader):
            inputs, labels = data
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)
            running_loss += loss
        avg_loss = running_loss / (i + 1)
        return avg_loss
