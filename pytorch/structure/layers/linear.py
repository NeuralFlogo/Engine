from torch import nn

import framework.structure.layers.linear


class PLinear:
    def __init__(self, linear: framework.structure.layers.linear.Linear):
        self.input_dimension = linear.input_dimension
        self.output_dimension = linear.output_dimension

    def build(self):
        return nn.Linear(in_features=self.input_dimension, out_features=self.output_dimension)

