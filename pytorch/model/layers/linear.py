from torch import nn


class Linear:
    def __init__(self, input_dimension, output_dimension):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

    def build(self):
        return nn.Linear(in_features=self.input_dimension, out_features=self.output_dimension)

