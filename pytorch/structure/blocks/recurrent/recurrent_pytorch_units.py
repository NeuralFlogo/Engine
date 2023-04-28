import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, activation="tanh"):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = getattr(nn, activation)()
        self.input_layer = nn.Linear(input_size, hidden_size, bias=bias)
        self.previous_layer = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, state=None):
        if state is None: state = Variable(x.new_zeros(x.size(0), self.hidden_size))
        return self.activation(self.input_layer(x.to(torch.float)) + self.previous_layer(state))


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.input_layer = nn.Linear(input_size, hidden_size * 4, bias=bias)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size * 4, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, state=None):
        if state is None:
            state = Variable(input.new_zeros(input.size(0), self.hidden_size))
            state = (state, state)
        state, memory = state
        gates = self.input_layer(input) + self.hidden_layer(state)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
        input_gate_out = torch.sigmoid(input_gate)
        forget_gate_out = torch.sigmoid(forget_gate)
        cell_gate_out = torch.tanh(cell_gate)
        output_gate_out = torch.sigmoid(output_gate)
        new_state = memory * forget_gate_out + input_gate_out * cell_gate_out
        cell_out = output_gate_out * torch.tanh(new_state)
        return cell_out, new_state


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.input_to_hidden = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.hidden_to_hidden = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def forward(self, x, state=None):
        if state is None:
            state = Variable(x.new_zeros(x.size(0), self.hidden_size))
        input_out = self.input_to_hidden(x)
        hidden_out = self.hidden_to_hidden(state)
        x_reset, x_upd, x_new = input_out.chunk(3, 1)
        h_reset, h_upd, h_new = hidden_out.chunk(3, 1)
        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))
        cell_output = update_gate * state + (1 - update_gate) * new_gate
        return cell_output
