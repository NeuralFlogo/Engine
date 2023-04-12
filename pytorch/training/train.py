import torch
from torch.utils.tensorboard import SummaryWriter

from pytorch.preprocesing.NumericProcessor import one_hot_encode


class Training:
    def __init__(self, epochs, model, training_loader, validation_loader, loss_function, optimizer):
        self.epochs = epochs
        self.model = model
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.writer = SummaryWriter("runs")

    def train(self):
        best_vloss = 1_000_000.
        for epoch in range(self.epochs):
            print('EPOCH {}:'.format(epoch + 1))
            self.model.train(True)
            avg_loss = self.__train_epoch(epoch)
            self.model.train(False)
            avg_vloss = self.__validate_epoch()
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            self.writer.add_scalars('Training vs Validation Loss',
                                    {'Training': avg_loss, 'Validation': avg_vloss},
                                    epoch + 1)
            print('Training vs Validation Loss',
                  {'Training': avg_loss, 'Validation': avg_vloss},
                  epoch + 1)
            self.writer.flush()
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                torch.save(self.model.state_dict(), 'models/model_{}'.format(epoch))
        self.writer.close()

    def __train_epoch(self, epoch):
        running_loss = 0.
        last_loss = 0.
        for i, data in enumerate(self.training_loader):
            inputs, labels = data
            labels = torch.tensor(one_hot_encode(labels), dtype=torch.float32)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            print('batch {} loss: {}'.format(i + 1, running_loss))
            tb_x = epoch * len(self.training_loader) + i + 1
            self.writer.add_scalar('Loss/train', running_loss, tb_x)
            last_loss = running_loss
            running_loss = 0.
        return last_loss

    def __validate_epoch(self):
        running_vloss = 0.
        for i, data in enumerate(self.validation_loader):
            inputs, labels = data
            labels = torch.tensor(one_hot_encode(labels))
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)
            running_vloss += loss.item()
        return running_vloss / (i + 1)
