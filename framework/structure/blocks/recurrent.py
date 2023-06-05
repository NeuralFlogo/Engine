class RecurrentBlock:
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, recurrent_unit: str):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.recurrent_unit = recurrent_unit

