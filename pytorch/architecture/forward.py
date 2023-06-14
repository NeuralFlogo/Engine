from torch.nn import Module, Sequential

from framework.structure.structure import Structure


class ForwardArchitecture(Module):
    def __init__(self, structure: Structure):
        super(ForwardArchitecture, self).__init__()
        self.metadata = structure.metadata
        self.architecture = Sequential()
        [self.architecture.append(i) for i in structure.content]

    def forward(self, x):
        return self.architecture(x)

    def to_device(self, device):
        self.to(device)

    def get_architecture(self):
        return self.architecture

    def get_section(self, index):
        return self.architecture[self.metadata.get_start_index(index):self.metadata.get_end_index(index)]

    def get_range(self, start, end):
        return self.architecture[self.metadata.get_start_index(start):self.metadata.get_end_index(end)]
