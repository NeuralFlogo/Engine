from torch.nn import Module, ModuleList


class RecurrentModule(Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = ModuleList(architecture)

    def forward(self, x):
        result = []
        state = None
        for block in self.architecture:
            block_output = block.forward(x, state=state)
            state = block_output
            result.append(state)
        return result


class LstmModel(Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = ModuleList(architecture)

    def forward(self, x):
        result = []
        state = None
        for block in self.architecture:
            block_output = block.forward(x, states=state)
            state = block_output[1]
            result.append(state)
        return result
