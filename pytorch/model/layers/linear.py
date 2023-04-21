from torch import nn

import model.flogo.layers.linear


class Linear:
    def __init__(self, linear: model.flogo.layers.linear.Linear):
        self.input_dimension = linear.input_dimension
        self.output_dimension = linear.output_dimension

    def build(self):
        return nn.Linear(in_features=self.input_dimension, out_features=self.output_dimension)

