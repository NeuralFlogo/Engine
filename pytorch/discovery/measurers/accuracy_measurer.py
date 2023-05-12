import torch


class AccuracyMeasurer:
    def measure(self, predictions, labels):
        return torch.sum(torch.eq(self.__predicted_class(predictions), self.__predicted_class(labels)))

    def __predicted_class(self, predictions):
        return torch.argmax(predictions, dim=1)
