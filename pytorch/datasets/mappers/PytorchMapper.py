import torch


class PytorchMapper:
    @staticmethod
    def map(data):
        return torch.tensor(data, dtype=torch.float32)

    @staticmethod
    def stack(data):
        return torch.stack(data)
