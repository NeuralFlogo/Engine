from torch import nn
import framework.structure.layers.linear


class PDropout:

    def __init__(self, dropout: framework.structure.layers.dropout.Dropout):
        self.probability = dropout.probability

    def build(self):
        return nn.Dropout(p=self.probability)
