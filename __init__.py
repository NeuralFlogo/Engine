import torch.nn

from flogo.structure.blocks.convolutional import ConvolutionalBlock
from flogo.structure.blocks.flatten import FlattenBlock
from flogo.structure.blocks.residual import ResidualBlock
from flogo.structure.layers.activation import Activation
from flogo.structure.layers.convolutional import Convolutional
from flogo.structure.layers.flatten import Flatten
from flogo.structure.layers.normalization import Normalization
from flogo.structure.layers.pool import Pool
from flogo.structure.sections.processing.convolutional import ConvolutionalSection
from flogo.structure.sections.processing.residual import ResidualSection
from flogo.structure.structure_factory import StructureFactory
from pytorch.preprocesing.SourceTypeFunctions import images_source_type
from pytorch.structure.generator import PytorchGenerator
from pytorch.structure.sections.link.flatten import FlattenSection

#
# feed_forward = [
#     FlogoLinearBlock(Flogo_layers.linear.Linear(100, 10), Flogo_layers.activation.Activation("ReLU")),
#     FlogoLinearBlock(Flogo_layers.linear.Linear(10, 2), Flogo_layers.activation.Activation("ReLU"))]
# FeedForwardSection(feed_forward).build()
#

#
# classification = FlogoClassificationBlock(Flogo_layers.classification.Classification("Softmax", 10))
# Classification(classification).build()

flatten = FlattenBlock(Flatten(1, 3))

# linear = [FlogoLinearBlock([architectures.flogo.layers.linear.Linear(156800, 1000),
#                            architectures.flogo.layers.activation.Activation("ReLU"),
#                            architectures.flogo.layers.linear.Linear(1000, 2)])]

# classification = FlogoClassificationBlock(architectures.flogo.layers.classification.Classification("Softmax", 1))

# residualSection = ResidualSection(residual).build()
flattenSection = FlattenSection(flatten).build()
# feedForwardSection = FeedForwardSection(linear).build()
# classificationSection = ClassificationSection(classification).build()

# train_data_loader, test_data_loader = images_source_type(226, 0, 1, "/Users/jose_juan/Desktop/training", 4)

# residualModule = ResidualModule(residualSection)
# flattenModule = ForwardModule(flattenSection)
# linearModule = ForwardModule(feedForwardSection)
# classificationModule = ForwardModule(classificationSection)
# architectures = CombinationModule(residualModule, flattenModule, linearModule, classificationModule)

# for i in train_data_loader:
#     print(architectures(i[0]))
#     break

# Training(FlogoTraining(20, architectures, train_data_loader, test_data_loader,
#          architectures.flogo.training.training.FlogoLossFunction("MSELoss"),
#          architectures.flogo.training.training.FlogoOptimizer("Adam", architectures.parameters(), 0.01))).train()


EPOCHS = 10
train_data_loader, test_data_loader = images_source_type(256, 0, 1,
                                                         "/Users/jose_juan/Desktop/monentia/training_set_test", 1)

convolutional1 = ConvolutionalSection([ConvolutionalBlock([Convolutional(3, 64, kernel=7, stride=2, padding=3),
                                                           Normalization(64), Activation("ReLU"),
                                                           Pool(kernel=3, stride=2, padding=1, pool_type="Max")]),
                                       ConvolutionalBlock([Convolutional(3, 64, kernel=7, stride=2, padding=3),
                                                           Normalization(64), Activation("ReLU"),
                                                           Pool(kernel=3, stride=2, padding=1, pool_type="Max")
                                                           ])])

residual = ResidualSection([ResidualBlock(64, 64, "ReLU", hidden_size=3),
                            ResidualBlock(64, 128, "ReLU", hidden_size=1, downsample=torch.nn.Sequential(
                                torch.nn.Conv2d(64, 128, (1, 1), (2, 2)),
                                torch.nn.BatchNorm2d(128))),
                            ResidualBlock(128, 128, "ReLU", hidden_size=3),
                            ResidualBlock(128, 256, "ReLU", hidden_size=1, downsample=torch.nn.Sequential(
                                torch.nn.Conv2d(128, 256, (1, 1), (2, 2)),
                                torch.nn.BatchNorm2d(256))),
                            ResidualBlock(256, 256, "ReLU", hidden_size=5),
                            ResidualBlock(256, 512, "ReLU", hidden_size=1, downsample=torch.nn.Sequential(
                                torch.nn.Conv2d(256, 512, (1, 1), (2, 2)),
                                torch.nn.BatchNorm2d(512))),
                            ResidualBlock(512, 512, "ReLU", hidden_size=3)])

structure = [convolutional1, residual]
print(StructureFactory(structure, PytorchGenerator()).create_structure())

# ForwardTraining(FlogoTraining(10, net, train_data_loader, test_data_loader, FlogoLossFunction("MSELoss"),
#                               FlogoOptimizer("Adam", [i for i in residualModule.parameters()] +
#                                              [i for i in feedForwardModule.parameters()], 0.01))).train()

# RecurrentTrain(FlogoTraining(2, lstmmodel, train_data_loader, test_data_loader,
#                              architectures.flogo.training.training.FlogoLossFunction("MSELoss"),
#                              architectures.flogo.training.training.FlogoOptimizer("Adam",
#                                                                           lstmmodel.parameters(), 0.01))).train()
