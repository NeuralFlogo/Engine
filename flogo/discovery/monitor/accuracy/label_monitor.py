import torch


class LabelMonitor:
    def compute(self, predictions, labels):
        return torch.sum(torch.eq(torch.argmax(predictions, dim=1), torch.argmax(labels, dim=1)))
