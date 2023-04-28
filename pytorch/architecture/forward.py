from torch.nn import Module, Sequential


class ForwardArchitecture(Module):
    def __init__(self, structure):
        super(ForwardArchitecture, self).__init__()
        self.architecture = Sequential()
        [self.architecture.append(i) for i in structure]

    def forward(self, x):
        return self.architecture(x)
