import unittest

from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, BatchNorm2d

from framework.structure.blocks.classification import ClassificationBlock
from framework.structure.blocks.convolutional import ConvolutionalBlock
from framework.structure.blocks.flatten import FlattenBlock
from framework.structure.blocks.linear import LinearBlock
from framework.structure.blocks.recurrent import RecurrentBlock
from framework.structure.blocks.residual import ResidualBlock
from framework.structure.layers.activation import Activation
from framework.structure.layers.classification import Classification
from framework.structure.layers.convolutional import Convolutional
from framework.structure.layers.flatten import Flatten
from framework.structure.layers.linear import Linear
from framework.structure.layers.pool import Pool
from pytorch.structure.blocks.convolutional import ConvolutionalBlock as PConvolutionalBlock
from pytorch.structure.blocks.recurrent import RecurrentBlock as PRecurrentBlock
from pytorch.structure.blocks.linear import LinearBlock as PLinearBlock
from pytorch.structure.blocks.flatten import FlattenBlock as PFlattenBlock
from pytorch.structure.blocks.classification import ClassificationBlock as PClassificationBlock


from pytorch.structure.blocks.residual import _ResidualBlock


class BlockTest(unittest.TestCase):

    def test_convolutional_block(self):
        block = ConvolutionalBlock([Convolutional(1, 6, kernel=5), Pool("Max"), Activation("ReLU")])
        expected = [Conv2d, MaxPool2d, ReLU]
        self.assertEqual(expected, [layer.__class__ for layer in PConvolutionalBlock(block).build()])

    def test_linear_block(self):
        block = LinearBlock([Linear(1600, 120), Activation("ReLU")])
        expected = [nn.Linear, ReLU]
        self.assertEqual(expected, [layer.__class__ for layer in PLinearBlock(block).build()])

    def test_residual_block(self):
        block = ResidualBlock(64, 64, "ReLU", hidden_size=3)
        torch_block = _ResidualBlock(block).build()
        layers = [layer for layer in torch_block.block1] + \
                 [layer for layer in torch_block.block2] + \
                 [torch_block.activation]
        expected = [Conv2d, BatchNorm2d, ReLU, Conv2d, BatchNorm2d, ReLU]
        self.assertEqual(expected, [layer.__class__ for layer in layers])

    def test_recurrent_block(self):
        block = RecurrentBlock(2, 200, 2, "RNN")
        torch_block = PRecurrentBlock(block).build()
        parameters = [torch_block.hidden_size, torch_block.num_layers, torch_block.training]
        expected = [200, 2, True]
        self.assertEqual(expected, parameters)

    def test_flatten_block(self):
        block = FlattenBlock(Flatten(1, 3))
        expected = nn.modules.flatten.Flatten
        self.assertEqual(expected, PFlattenBlock(block).build().__class__)

    def test_classification_block(self):
        block = ClassificationBlock(Classification("Softmax", 2))
        expected = nn.modules.activation.Softmax
        self.assertEqual(expected, PClassificationBlock(block).build().__class__)
