import unittest

from torch import nn
from torch.nn import modules, Softmax

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
from framework.structure.sections.link.classificationsection import ClassificationSection
from framework.structure.sections.link.flatten import FlattenSection
from framework.structure.sections.processing.convolutional import ConvolutionalSection
from framework.structure.sections.processing.linear import LinearSection
from framework.structure.sections.processing.recurrent import RecurrentSection
from framework.structure.sections.processing.residual import ResidualSection
from pytorch.structure.blocks.recurrent import Block
from pytorch.structure.blocks.residual import ResidualBlock as PResidualBlock
from pytorch.structure.sections.link.classification import ClassificationSection as PClassificationSection
from pytorch.structure.sections.link.flatten import FlattenSection as PFlattenSection
from pytorch.structure.sections.processing.convolutional import ConvolutionalSection as PConvolutionalSection
from pytorch.structure.sections.processing.linear import LinearSection as PLinearSection
from pytorch.structure.sections.processing.recurrent import RecurrentSection as PRecurrentSection
from pytorch.structure.sections.processing.residual import ResidualSection as PResidualSection


class SectionsTest(unittest.TestCase):

    def test_linear_section(self):
        section = LinearSection([LinearBlock([Linear(1600, 120), Activation("ReLU")]), LinearBlock([Linear(120, 10)])])
        expected = [modules.linear.Linear, modules.activation.ReLU, modules.linear.Linear]
        self.assertEqual(expected, [layer.__class__ for layer in PLinearSection(section).build()])

    def test_convolution_section(self):
        section = ConvolutionalSection([ConvolutionalBlock([Convolutional(1, 6, kernel=5), Pool("Max")])])
        expected = [modules.conv.Conv2d, modules.pooling.MaxPool2d]
        self.assertEqual(expected, [layer.__class__ for layer in PConvolutionalSection(section).build()])

    def test_residual_section(self):
        section = ResidualSection([ResidualBlock(64, 64, "ReLU"), ResidualBlock(64, 64, "ReLU")])
        expected = [PResidualBlock, PResidualBlock]
        self.assertEqual(expected, [layer.__class__ for layer in PResidualSection(section).build()])

    def test_recurrent_section(self):
        section = RecurrentSection([RecurrentBlock(4, 200, 2, "RNN")])
        expected = [Block]
        self.assertEqual(expected, [layer.__class__ for layer in PRecurrentSection(section).build()])

    def test_classification_section(self):
        section = ClassificationSection(ClassificationBlock(Classification("Softmax", 1)))
        expected = [Softmax]
        self.assertEqual(expected, [layer.__class__ for layer in PClassificationSection(section).build()])

    def test_flatten_section(self):
        section = FlattenSection(FlattenBlock(Flatten(3, 1)))
        expected = [nn.Flatten]
        self.assertEqual(expected, [layer.__class__ for layer in PFlattenSection(section).build()])
