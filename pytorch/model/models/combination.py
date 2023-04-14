from torch.nn import Module


class CombinationModule(Module):
    def __init__(self, *models: Module):
        super().__init__()
        self.models = models

    def forward(self, x):
        for model in self.models:
            x = model.forward(x)
        return x

