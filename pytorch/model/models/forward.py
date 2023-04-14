from torch.nn import Module, Sequential


class ForwardModule(Module):
    def __init__(self, architecture):
        super(ForwardModule, self).__init__()
        self.architecture = Sequential()
        [self.architecture.append(i) for i in architecture]

    def forward(self, x):
        return self.architecture(x)
