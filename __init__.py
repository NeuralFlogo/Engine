import torch

import model.flogo.blocks.convolutional
from model.flogo.blocks.classification import FlogoClassificationBlock
from model.flogo.blocks.flatten import FlogoFlattenBlock
from model.flogo.blocks.linear import FlogoLinearBlock
from model.flogo.blocks.recurrent import FlogoRecurrentBlock
from model.flogo.blocks.residual import FlogoOutputBlock, FlogoBodyBlock, FlogoInputBlock
from model.flogo.layers import pool
from model.flogo.layers.flatten import Flatten
from pytorch.model.models.combination import CombinationModule
from pytorch.model.models.forward import ForwardModule
from pytorch.model.models.recurrent import RecurrentModule, LstmModule
from pytorch.model.models.residual import ResidualModule
from pytorch.model.sections.link.classification import ClassificationSection
from pytorch.model.sections.link.flatten import FlattenSection
from pytorch.model.sections.processing.feed_forward import FeedForwardSection
from pytorch.model.sections.processing.recurrent.recurrent import RecurrentSection
from pytorch.model.sections.processing.residual import ResidualSection
from pytorch.preprocesing.SourceTypeFunctions import images_source_type

#
# feed_forward = [
#     FlogoLinearBlock(Flogo_layers.linear.Linear(100, 10), Flogo_layers.activation.Activation("ReLU")),
#     FlogoLinearBlock(Flogo_layers.linear.Linear(10, 2), Flogo_layers.activation.Activation("ReLU"))]
# FeedForwardSection(feed_forward).build()
#
# flatten = FlogoFlattenBlock(Flogo_layers.flatten.Flatten(10, 8))
# Flatten(flatten).build()
#
# classification = FlogoClassificationBlock(Flogo_layers.classification.Classification("Softmax", 10))
# Classification(classification).build()

# residual = [FlogoInputBlock(model.flogo.layers.convolutional.Conv(channel_in=3, channel_out=64, kernel=7),
#                            model.flogo.layers.pool.Pool(pool_type="Max", kernel=7)),
#            FlogoBodyBlock(content=[model.flogo.layers.convolutional.Conv(channel_in=64, channel_out=128, kernel=3),
#                                    model.flogo.layers.convolutional.Conv(channel_in=128, channel_out=128, kernel=3),
#                                    model.flogo.layers.convolutional.Conv(channel_in=128, channel_out=64, kernel=3)],
#                           hidden_size=3),
#            FlogoBodyBlock(content=[model.flogo.layers.convolutional.Conv(channel_in=128, channel_out=256, kernel=3),
#                                    model.flogo.layers.convolutional.Conv(channel_in=256, channel_out=256, kernel=3),
#                                    model.flogo.layers.convolutional.Conv(channel_in=256, channel_out=128, kernel=3)],
#                           hidden_size=3),
#            FlogoOutputBlock(model.flogo.layers.pool.Pool(pool_type="Avg"))]

# flatten = FlogoFlattenBlock(Flatten(1, 3))

# linear = [FlogoLinearBlock([model.flogo.layers.linear.Linear(156800, 1000),
#                            model.flogo.layers.activation.Activation("ReLU"),
#                            model.flogo.layers.linear.Linear(1000, 2)])]

# classification = FlogoClassificationBlock(model.flogo.layers.classification.Classification("Softmax", 1))

# residualSection = ResidualSection(residual).build()
# flattenSection = FlattenSection(flatten).build()
# feedForwardSection = FeedForwardSection(linear).build()
# classificationSection = ClassificationSection(classification).build()

# train_data_loader, test_data_loader = images_source_type(226, 0, 1, "/Users/jose_juan/Desktop/training", 4)

# residualModule = ResidualModule(residualSection)
# flattenModule = ForwardModule(flattenSection)
# linearModule = ForwardModule(feedForwardSection)
# classificationModule = ForwardModule(classificationSection)
# model = CombinationModule(residualModule, flattenModule, linearModule, classificationModule)

# for i in train_data_loader:
#     print(model(i[0]))
#     break

# Training(FlogoTraining(20, model, train_data_loader, test_data_loader,
#          model.flogo.training.training.FlogoLossFunction("MSELoss"),
#          model.flogo.training.training.FlogoOptimizer("Adam", model.parameters(), 0.01))).train()


recurrent = [FlogoRecurrentBlock(3, 1, 2, "LSTMCell", "ReLU", True)]
recurrentBuild = RecurrentSection(recurrent).build()
lstmModule = LstmModule(recurrentBuild)

print(lstmModule.forward([torch.ones(3), torch.ones(3)]))
