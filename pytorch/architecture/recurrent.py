from torch.nn import Module, ModuleList


class RecurrentArchitecture(Module):
    def __init__(self, structure):
        super().__init__()
        self.architecture = ModuleList(structure)

    def forward(self, x: list):
        result = []
        state = None
        for i, block in enumerate(self.architecture):
            block_output = block.forward(x[i], state=state)
            state = block_output
            result.append(block_output)
        return result


class LstmArchitecture(Module):
    def __init__(self, structure):
        super().__init__()
        self.architecture = ModuleList(structure)

    def forward(self, x: list):
        result = []
        state = None
        for i, block in enumerate(self.architecture):
            block_output = block.forward(x[i], states=state)
            state = block_output[1]
            result.append(block_output[0])
        return result
