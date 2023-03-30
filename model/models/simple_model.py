import torch.nn


class SimpleModel(torch.nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.layers = torch.nn.ModuleList(architecture)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
