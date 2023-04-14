import numpy as np
import torch
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = getattr(nn, nonlinearity)()
        self.input_layer = nn.Linear(input_size, hidden_size, bias=bias)
        self.previous_layer = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, state=None):
        if state is None: state = torch.zeros(self.hidden_size)
        return self.activation(self.input_layer(input.to(torch.float)) + self.previous_layer(state))


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.input_forget_layer = nn.Linear(input_size, hidden_size, bias=bias)
        self.state_forget_layer = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.input_input_layer = nn.Linear(input_size, hidden_size, bias=bias)
        self.state_input_layer = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.input_output_gate = nn.Linear(input_size, hidden_size, bias=bias)
        self.state_output_gate = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def forward(self, x, states=None):
        if states is None: states = self.initialize()
        current_long_term_state = (self.forget_gate(x, states) * states[0]) + self.input_gate(x, states)
        output_gate_output = self.output_gate(current_long_term_state, x, states)
        return output_gate_output, [current_long_term_state, output_gate_output]

    def forget_gate(self, x, states):
        return self.sigmoid(self.input_forget_layer(x) + self.state_forget_layer(states[1]))

    def input_gate(self, x, states):
        sigmoid_input_gate_output = self.sigmoid(self.input_input_layer(x) + self.state_input_layer(states[1]))
        input_gate_output = sigmoid_input_gate_output * self.tanh(states[1])
        return input_gate_output

    def output_gate(self, current_long_term_state, x, states):
        sigmoid_output_gate = self.sigmoid(self.input_output_gate(x) + self.state_output_gate(states[1]))
        output_gate_output = sigmoid_output_gate * self.tanh(current_long_term_state)
        return output_gate_output

    def initialize(self):
        return [torch.zeros(self.hidden_size, dtype=torch.float),
                torch.zeros(self.hidden_size, dtype=torch.float)]


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.input_reset_layer = nn.Linear(input_size, hidden_size, bias=bias)
        self.state_reset_layer = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.input_update_layer = nn.Linear(input_size, hidden_size, bias=bias)
        self.state_update_layer = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.input_candidate_layer = nn.Linear(input_size, hidden_size, bias=bias)
        self.state_candidate_layer = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, state=None):
        if state is None: state = self.initialize()
        update_gate_output = self.update_gate(input, state)
        candidate_gate_output = self.candidate_gate(input, self.reset_gate(input, state), state)
        return update_gate_output * state + (1 - update_gate_output) * candidate_gate_output

    def reset_gate(self, input, state):
        return self.sigmoid(self.input_reset_layer(input) + self.state_reset_layer(state))

    def update_gate(self, input, state):
        return self.sigmoid(self.input_update_layer(input) + self.state_update_layer(state))

    def candidate_gate(self, input, reset_gate_output, state):
        return self.tanh(self.input_candidate_layer(input) + self.state_candidate_layer(reset_gate_output * state))

    def initialize(self):
        return torch.zeros(self.hidden_size, dtype=torch.float)
