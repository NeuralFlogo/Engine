import torch.nn

from flogo.structure.blocks import recurrent


class RecurrentBlock:
    def __init__(self, block: recurrent.RecurrentBlock):
        self.input_size = block.input_size
        self.hidden_size = block.hidden_size
        self.recurrent_unit = block.recurrent_unit
        self.num_layers = block.num_layers

    def build(self):
        if self.recurrent_unit == "RNN":
            return Block(self.input_size, self.hidden_size, self.num_layers, torch.nn.RNN)
        if self.recurrent_unit == "GRU":
            return Block(self.input_size, self.hidden_size, self.num_layers, torch.nn.GRU)
        if self.recurrent_unit == "LSTM":
            return Block(self.input_size, self.hidden_size, self.num_layers, torch.nn.LSTM)


class Block(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, unit):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.unit = unit(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        hidden_state = self.initialize_state()
        out, _ = self.unit(x, (hidden_state, self.initialize_state())) if self.isLSTM() else self.unit(x, hidden_state)
        return out.reshape(out.shape[0], -1)

    def isLSTM(self):
        return type(self.unit) == torch.nn.LSTM

    def initialize_state(self):
        return torch.zeros(self.num_layers, self.hidden_size)
