import torch
from jinja2.nodes import Test

import model.flogo.layers as layers
from model.flogo.blocks.classification import FlogoClassificationBlock
from model.flogo.blocks.convolutional import FlogoConvolutionalBlock
from model.flogo.blocks.flatten import FlogoFlattenBlock
from model.flogo.blocks.linear import FlogoLinearBlock
from model.flogo.layers.linear import Linear
from pytorch.model.models.simple_model import SimpleModel
from pytorch.model.sections.link.classification import ClassificationSection
from pytorch.model.sections.link.flatten import FlattenSection

from pytorch.model.sections.processing.convolutional import ConvolutionalSection
from pytorch.model.sections.processing.feed_forward import FeedForwardSection
from pytorch.preprocesing.SourceTypeFunctions import images_source_type
from pytorch.training.test import Testing
from pytorch.training.train import Training

#
# feed_forward = [
#     CompiledLinearBlock(compiled_layers.linear.Linear(100, 10), compiled_layers.activation.Activation("ReLU")),
#     CompiledLinearBlock(compiled_layers.linear.Linear(10, 2), compiled_layers.activation.Activation("ReLU"))]
# FeedForwardSection(feed_forward).build()
#
# flatten = CompiledFlattenBlock(compiled_layers.flatten.Flatten(10, 8))
# Flatten(flatten).build()
#
# classification = CompiledClassificationBlock(compiled_layers.classification.Classification("Softmax", 10))
# Classification(classification).build()

convolutional = [FlogoConvolutionalBlock([layers.convolutional.Conv(3, 20, stride=1, padding=0, kernel=5),
                                          layers.activation.Activation("ReLU"),
                                          layers.pool.Pool("Max"),
                                          layers.convolutional.Conv(20, 40, stride=1, padding=0, kernel=5),
                                          layers.activation.Activation("ReLU"),
                                          layers.pool.Pool("Max")])]

flatten = FlogoFlattenBlock(layers.flatten.Flatten(1, 3))

linear = [FlogoLinearBlock([Linear(3240, 1000),
                            layers.activation.Activation("ReLU"),
                            layers.linear.Linear(1000, 1000),
                            layers.activation.Activation("ReLU"),
                            layers.linear.Linear(1000, 2)]
                           )]

classification = FlogoClassificationBlock(layers.classification.Classification("Softmax", 1))

convolutional_section = ConvolutionalSection(convolutional).build()
flatten_section = FlattenSection(flatten).build()
linear_section = FeedForwardSection(linear).build()
classification_section = ClassificationSection(classification).build()

model = SimpleModel(convolutional_section + [flatten_section] + linear_section)


train_data_loader, test_data_loader = images_source_type(50, 0, 1, "C:/Users/Joel/Desktop/prueba_images/training_set", 47)
Training(2, model, train_data_loader, test_data_loader,
         torch.nn.MSELoss(),
         torch.optim.Adam(model.parameters(), lr=0.1)).train()

Testing(model, test_data_loader).test()

# residual = [CompiledInputBlock(compiled_layers.convolutional.Conv((), 10, 10, (), ()),
#                                compiled_layers.pool.Pool((), (), (), "Max")),
#             CompiledBodyBlock([compiled_layers.convolutional.Conv((), 10, 10, (), ()),
#                                  compiled_layers.activation.Activation("ReLU"),
#                                  compiled_layers.convolutional.Conv((), 10, 10, (), ())], 2),
#             CompiledOutputBlock(compiled_layers.pool.Pool((), (), (), "Max"))]
# ResidualSection(residual).build()
#
# recurrent = [CompiledRecurrentBlock(3, 15, 5, "GRUCell", "ReLU", True)]
# RecurrentSection(recurrent).build()
