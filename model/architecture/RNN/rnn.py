from model.architecture.RNN.recurrent_pytorch_units import LSTMCell, RNNCell, GRUCell
from vocabulary import Channel, Block, Activation, Layers


class RNN:
    def __init__(self, architecture):
        self.architecture = []
        for i in range(architecture[Layers.Size]):
            self.architecture.append(RecurrentBlock(architecture[Channel.In], architecture[Block.HiddenSize],
                                                    architecture[Block.Type], architecture[Activation.name],
                                                    architecture[Layers.Bias]))

    def pytorch(self):
        return [block.pytorch() for block in self.architecture]


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
