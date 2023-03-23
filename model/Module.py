import torch.nn


class Module(torch.nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.layers = torch.nn.ModuleList(architecture)

    def forward(self, x):
        for layer in self.architecture:
            x = layer(x)
        return x
