class FlogoRecurrentBlock:
    def __init__(self, channel_in: int, channel_out: int, hidden_size: int, type_: str, activation_name: str, bias: bool):
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.hidden_size = hidden_size
        self.type_ = type_
        self.activation_name = activation_name
        self.bias = bias
