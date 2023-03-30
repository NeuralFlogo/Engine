import torch


class ResidualModule(torch.nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.input_block = torch.nn.ModuleList(list(architecture[0]))
        self.blocks = []
        for block in architecture[1]:
            self.blocks.append(torch.nn.ModuleList(list(block)))
        self.output_block = torch.nn.ModuleList([architecture[2]])

    def forward(self, x):
        x = self.input_forward(x)
        x = self.residual_forward(x)
        return self.output_forward(x)

    def input_forward(self, x):
        for layer in self.input_block:
            x = layer(x)
        return x

    def residual_forward(self, x):
        for block in self.blocks:
            residual = x
            for layer in block:
                if not isinstance(layer, torch.nn.ReLU): residual = layer(residual)
                x = layer(x)
            x = residual + x
        return x

    def output_forward(self, x):
        for layer in self.output_block:
            x = layer(x)
        return x
