import torch


class LossMeasurer:
    def measure(self, predictions, expected):
        return torch.sum(torch.abs(expected - predictions)).item()