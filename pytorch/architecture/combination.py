from torch.nn import Module


class CombinationArchitecture(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        self.modules = modules

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x

