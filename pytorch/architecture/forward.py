from torch.nn import Module, Sequential

from framework.structure.runnable import Runnable


class ForwardArchitecture(Module):
    def __init__(self, runnable: Runnable):
        super(ForwardArchitecture, self).__init__()
        self.metadata = runnable.metadata
        self.architecture = Sequential()
        [self.architecture.append(item) for item in runnable.structure]

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
