import torch
from torch.nn import Module, ModuleList, Sequential


class ResidualModule(Module):
    def __init__(self, architecture):
        super().__init__()
        self.input_block = ModuleList(architecture[0])
        self.body_stages = torch.nn.ModuleList()
        for stage in architecture[1:-1]:
            for block in stage: 
                self.body_stages.append(ModuleList(block))
        self.output_block = Sequential(architecture[-1])

    def forward(self, x):
        x = self.input_forward(x)
        x = self.residual_forward(x)
        return self.output_forward(x)

    def input_forward(self, x):
        for layer in self.input_block:
            x = layer(x)
        return x

    def residual_forward(self, x):
        for i, block in enumerate(self.body_stages):
            residual = x
            if self.body_stages[i][0].in_channels != self.body_stages[i-1][0].in_channels if i-1 != -1 else False:
                x = torch.cat((residual, x), 1)
            for layer in block:
                x = layer(x)
        return x

    def output_forward(self, x):
        for layer in self.output_block:
            x = layer(x)
        return x
