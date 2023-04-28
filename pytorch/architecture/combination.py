from torch.nn import Module


class CombinationArchitecture(Module):
    def __init__(self, *architectures: Module):
        super().__init__()
        self.architecture = architectures

    def forward(self, x):
        for architecture in self.architecture:
            x = architecture.forward(x)
        return x

