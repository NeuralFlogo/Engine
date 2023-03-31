import torch.nn


class SimpleModel(torch.nn.Module):
    def __init__(self, architecture):
        super(SimpleModel, self).__init__()
        self.module_list = torch.nn.Sequential()
        [self.module_list.append(i) for i in architecture]

    def forward(self, x):
        x = self.module_list(x)
        return x
