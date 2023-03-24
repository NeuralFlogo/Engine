import torch.nn
from torch.autograd import Variable


class SimpleModel(torch.nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.layers = torch.nn.ModuleList(architecture)

    def forward(self, x):
        for layer in self.architecture:
            x = layer(x)
        return x


class SimpleRNN(torch.nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.layers = torch.nn.ModuleList(architecture)

    def forward(self, x, hx=None):
        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.layers[1].hidden_size()).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.layers[1].hidden_size()))
        else:
            h0 = hx
        outs = []
        hidden = list()
        for layer in range(len(self.layers)):
            hidden.append(h0[layer, :, :])
        for t in range(input.size(1)):
            for layer in range(len(self.layers)):
                if layer == 0:
                    hidden_l = self.layers[layer](input[:, t, :], hidden[layer])
                else:
                    hidden_l = self.layers[layer](hidden[layer - 1], hidden[layer])
                hidden[layer] = hidden_l
                hidden[layer] = hidden_l
            outs.append(hidden_l)
        out = outs[-1].squeeze()
        out = self.layers[-1](out)
        return out


class LSTM(torch.nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.layers = torch.nn.ModuleList(architecture)

    def forward(self, x, hx=None):
        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(len(self.layers), x.size(0), self.layers[1].hidden_size()).cuda())
            else:
                h0 = Variable(torch.zeros(len(self.layers), x.size(0), self.layers[1].hidden_size()))
        else:
            h0 = hx
        outs = []
        hidden = list()
        for layer in range(len(self.layers)):
            hidden.append((h0[layer, :, :], h0[layer, :, :]))
        for t in range(x.size(1)):
            for layer in range(len(self.layers)):
                if layer == 0:
                    hidden_l = self.layers[layer](
                        x[:, t, :],
                        (hidden[layer][0], hidden[layer][1])
                    )
                else:
                    hidden_l = self.layers[layer](
                        hidden[layer - 1][0],
                        (hidden[layer][0], hidden[layer][1])
                    )
                hidden[layer] = hidden_l
            outs.append(hidden_l[0])
        out = outs[-1].squeeze()
        out = self.layers[-1](out)
        return out


class GRU(torch.nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.layers = torch.nn.ModuleList(architecture)

    def forward(self, x, hx=None):
        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.layers[1].hidden_size()).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.layers[1].hidden_size()))
        else:
            h0 = hx
        outs = []
        hidden = list()
        for layer in range(len(self.layers)):
            hidden.append(h0[layer, :, :])
        for t in range(x.size(1)):
            for layer in range(len(self.layers)):
                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](x[:, t, :], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1], hidden[layer])
                hidden[layer] = hidden_l
                hidden[layer] = hidden_l
            outs.append(hidden_l)
        out = outs[-1].squeeze()
        out = self.fc(out)
        return out
