from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer
from flogo.discovery.regularization.early_stopping import EarlyStopping


class ForwardTrainer:
    def __init__(self, training_dataset, validation_dataset, loss_function: Loss, optimizer: Optimizer, accuracy_monitor, early_stopping: EarlyStopping):
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.accuracy_monitor = accuracy_monitor
        self.early_stopping = early_stopping

    def train(self, epochs, model):
        for epoch in range(epochs):
            model.train(True)
            loss = self.__train_model(epochs, model)
            model.train(False)
            vloss, hit = self.__validate_model(epochs, model)
            self.__log_epoch_losses(epoch, loss, vloss)
            self.__log_epoch_accuracy(epoch, hit)
            if not self.early_stopping.check(self.__to_percentage(hit, len(self.validation_dataset)), vloss):
                return model
        return model

    def __train_model(self, epochs, model):
        running_loss = 0.
        for i, entry in enumerate(self.training_dataset, start=1):
            inputs, labels = entry.get_input(), entry.get_output()
            predictions = self.__evaluate(model, inputs)
            loss = self.loss_function.compute(predictions, labels)
            loss.backward()
            running_loss += loss.item()
            self.optimizer.step()
        return self.__epoch_average_loss(epochs, running_loss)

    def __validate_model(self, epochs, model):
        running_loss = 0.
        hit = 0
        for i, entry in enumerate(self.validation_dataset, start=1):
            inputs, labels = entry.get_input(), entry.get_output()
            predictions = self.__evaluate(model, inputs)
            running_loss += self.loss_function.compute(predictions, labels).item()
            hit += self.accuracy_monitor.compute(predictions, labels)
        return self.__epoch_average_loss(epochs, running_loss), hit

    def __evaluate(self, model, inputs):
        return model(inputs)

    def __log_epoch_losses(self, epoch, train_loss, val_loss):
        print('Epoch {} Training - Validation Loss: Training = {}, Validation = {}'.format(epoch + 1, train_loss,
                                                                                           val_loss))

    def __epoch_average_loss(self, epochs, loss):
        return loss / epochs

    def __to_percentage(self, value, size):
        return (100 * value / size).item()

    def __log_epoch_accuracy(self, epoch, correct):
        print('Epoch {} Accuracy: {}/{} ({:.0f}%)'
              .format(epoch + 1, correct, len(self.validation_dataset),
                      self.__to_percentage(correct, len(self.validation_dataset))))
