import torch

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

convolutional = [FlogoConvolutionalBlock([layers.convolutional.Conv(3, 64, stride=1, padding=1, kernel=3),
                                          layers.activation.Activation("ReLU"),
                                          layers.pool.Pool("Max"),
                                          layers.convolutional.Conv(64, 128, stride=1, padding=1, kernel=3),
                                          layers.activation.Activation("ReLU"),
                                          layers.pool.Pool("Max"),
                                          layers.convolutional.Conv(128, 256, stride=1, padding=1, kernel=3),
                                          layers.activation.Activation("ReLU"),
                                          layers.pool.Pool("Max"),
                                          layers.pool.Pool("Max"),
                                          layers.convolutional.Conv(256, 512, stride=1, padding=1, kernel=3),
                                          layers.activation.Activation("ReLU"),
                                          layers.pool.Pool("Max"),
                                          layers.convolutional.Conv(512, 512, stride=1, padding=1, kernel=3),
                                          layers.activation.Activation("ReLU"),
                                          layers.pool.Pool("Max")])]

flatten = FlogoFlattenBlock(layers.flatten.Flatten(1, 3))

linear = [FlogoLinearBlock([Linear(4608, 4096),
                            layers.activation.Activation("ReLU"),
                            layers.linear.Linear(4096, 4096),
                            layers.activation.Activation("ReLU"),
                            layers.linear.Linear(4096, 2)]
                           )]

classification = FlogoClassificationBlock(layers.classification.Classification("Softmax", 1))

convolutional_section = ConvolutionalSection(convolutional).build()
flatten_section = FlattenSection(flatten).build()
linear_section = FeedForwardSection(linear).build()
classification_section = ClassificationSection(classification).build()

model = SimpleModel(convolutional_section + [flatten_section] + linear_section)

print(model)

train_data_loader, test_data_loader = images_source_type(226, 0, 1, "/Users/jose_juan/Desktop/training", 2)
# Training(20, model, train_data_loader, test_data_loader,
#          torch.nn.MSELoss(),
#          torch.optim.Adam(model.parameters(), lr=0.001)).train()

sequential = torch.nn.Sequential(torch.load("/Users/jose_juan/PycharmProjects/Flogo/models/model_18"))
print([sequential(i) for i in train_data_loader])


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
