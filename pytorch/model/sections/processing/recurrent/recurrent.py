from compiled.model.blocks.recurrent import CompiledRecurrentBlock
from pytorch.model.sections.processing.recurrent.recurrent_pytorch_units import LSTMCell, RNNCell, GRUCell


class RecurrentSection:
    def __init__(self, section):
        self.architecture = []
        for block in section:
            self.architecture += (RecurrentBlock(block) for _ in range(block.hidden_size))

    def build(self):
        return [block.build() for block in self.architecture]


class RecurrentBlock:
    def __init__(self, block: CompiledRecurrentBlock):
        self.channel_in = block.channel_in
        self.channel_out = block.channel_out
        self.block_type = block.type_
        self.activation = block.activation_name
        self.bias = block.bias

    def build(self):
        if self.block_type == "LSTMCell": return LSTMCell(self.channel_in, self.channel_out, self.bias)
        if self.block_type == "RNNCell": return RNNCell(self.channel_in, self.channel_out, self.bias, self.activation)
        if self.block_type == "GRUCell": return GRUCell(self.channel_in, self.channel_out, self.bias)
