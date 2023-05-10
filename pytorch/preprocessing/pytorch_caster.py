import torch


class PytorchCaster:

    @staticmethod
    def cast(values):
        print(values)
        print(type(values[0]))
        return torch.tensor(values, dtype=type(values[0]))