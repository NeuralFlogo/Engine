class CompiledRecurrentBlock:
    def __init__(self, channel_in, channel_out, hidden_size, type_, name, bias):
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.hidden_size = hidden_size
        self.type_ = type_
        self.name = name
        self.bias = bias
