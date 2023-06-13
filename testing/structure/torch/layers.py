import unittest

from torch import nn
from torch.nn import ReLU, Softmax, Conv2d

from framework.structure.layers.activation import Activation
from framework.structure.layers.classification import Classification
from framework.structure.layers.convolutional import Convolutional
from framework.structure.layers.dropout import Dropout
from framework.structure.layers.flatten import Flatten
from framework.structure.layers.linear import Linear
from framework.structure.layers.normalization import Normalization
from framework.structure.layers.pool import Pool
from pytorch.structure.layers.activation import PActivation
from pytorch.structure.layers.classification import PClassification
from pytorch.structure.layers.convolution import PConvolutional
from pytorch.structure.layers.dropout import PDropout
from pytorch.structure.layers.flatten import PFlatten
from pytorch.structure.layers.linear import PLinear
from pytorch.structure.layers.normalization import PNormalization
from pytorch.structure.layers.pool import PPool


class LayersTest(unittest.TestCase):

    def test_activation_layer(self):
        layer = PActivation(Activation("ReLU"))
        expected = ReLU
        self.assertEqual(expected, layer.build().__class__)

    def test_classification_layer(self):
        layer = PClassification(Classification("Softmax", 2))
        expected = Softmax
        self.assertEqual(expected, layer.build().__class__)

    def test_convolution_layer(self):
        layer = PConvolutional(Convolutional(3, 16))
        expected = Conv2d
        self.assertEqual(expected, layer.build().__class__)

    def test_flatten_layer(self):
        layer = PFlatten(Flatten(3, 1))
        expected = nn.Flatten
        self.assertEqual(expected, layer.build().__class__)

    def test_linear_layer(self):
        layer = PLinear(Linear(20, 40))
        expected = nn.Linear
        self.assertEqual(expected, layer.build().__class__)

    def test_normalization_layer(self):
        layer = PNormalization(Normalization(10))
        expected = nn.BatchNorm2d
        self.assertEqual(expected, layer.build().__class__)

    def test_pool_layer(self):
        layer = PPool(Pool("Max"))
        expected = nn.MaxPool2d
        self.assertEqual(expected, layer.build().__class__)

    def test_dropout_layer(self):
        layer = PDropout(Dropout(0.2))
        expected = nn.Dropout
        self.assertEqual(expected, layer.build().__class__)
