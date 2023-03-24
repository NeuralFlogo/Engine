from torch import nn

from model.RNN.RecurrentPytorchCells import LSTMCell, RNNCell, GRUCell


class Recurrent:
    def __init__(self, architecture):
        self.architecture = []
        for i in range(architecture["num_layers"]):
            if i == 0:
                self.architecture.append(RecurrentBlock(architecture["input_size"], architecture["hidden_size"],
                                                        architecture["block_type"], architecture["activation"],
                                                        architecture["bias"]))
            self.architecture.append(RecurrentBlock(architecture["hidden_size"], architecture["hidden_size"],
                                                    architecture["block_type"], architecture["activation"],
                                                    architecture["bias"]))
        self.architecture.append(LinearFC(architecture["hidden_size"], architecture["output_size"]))

    def pytorch(self):
        result = []
        for block in self.architecture: result.append(block.pytorch())
        return result


class RecurrentBlock:
    def __init__(self, input_size, output_size, block_type, activation, bias):
        self.input_size = input_size
        self.output_size = output_size
        self.block_type = block_type
        self.activation = activation
        self.bias = bias

    def pytorch(self):
        if self.block_type == "LSTMCell": return LSTMCell(self.input_size, self.output_size, self.bias)
        if self.block_type == "RNNCell": return RNNCell(self.input_size, self.output_size, self.bias, self.activation)
        if self.block_type == "GRUCell": return GRUCell(self.input_size, self.output_size, self.bias)


class LinearFC:
    def __init__(self, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.output_size = output_size

    def pytorch(self):
        return nn.Linear(self.hidden_size, self.output_size)

