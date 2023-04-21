import torch.nn

from model.flogo.blocks.convolutional import FlogoConvolutionalBlock
from model.flogo.blocks.flatten import FlogoFlattenBlock
from model.flogo.blocks.linear import FlogoLinearBlock
from model.flogo.blocks.residual import FlogoResidualBlock
from model.flogo.layers.activation import Activation
from model.flogo.layers.convolutional import Conv
from model.flogo.layers.flatten import Flatten
from model.flogo.layers.linear import Linear
from model.flogo.layers.normalization import Normalization
from model.flogo.layers.pool import Pool
from pytorch.model.sections.link.flatten import FlattenSection
from pytorch.model.sections.processing.convolutional import ConvolutionalSection
from pytorch.model.sections.processing.feed_forward import FeedForwardSection
from pytorch.model.sections.processing.residual import ResidualSection
from pytorch.preprocesing.SourceTypeFunctions import images_source_type

#
# feed_forward = [
#     FlogoLinearBlock(Flogo_layers.linear.Linear(100, 10), Flogo_layers.activation.Activation("ReLU")),
#     FlogoLinearBlock(Flogo_layers.linear.Linear(10, 2), Flogo_layers.activation.Activation("ReLU"))]
# FeedForwardSection(feed_forward).build()
#

#
# classification = FlogoClassificationBlock(Flogo_layers.classification.Classification("Softmax", 10))
# Classification(classification).build()

flatten = FlogoFlattenBlock(Flatten(1, 3))

# linear = [FlogoLinearBlock([model.flogo.layers.linear.Linear(156800, 1000),
#                            model.flogo.layers.activation.Activation("ReLU"),
#                            model.flogo.layers.linear.Linear(1000, 2)])]

# classification = FlogoClassificationBlock(model.flogo.layers.classification.Classification("Softmax", 1))

# residualSection = ResidualSection(residual).build()
flattenSection = FlattenSection(flatten).build()
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


EPOCHS = 200
parameters = ["one-hot"] * 22
train_data_loader, test_data_loader = images_source_type(256, 0, 1, "/Users/jose_juan/Desktop/training", 1)

convolutional1 = [FlogoConvolutionalBlock([Conv(3, 64, kernel=7, stride=2, padding=3), Normalization(64),
                                           Activation("ReLU"), Pool(kernel=3, stride=2, padding=1, pool_type="Max")])]

residual = [FlogoResidualBlock(64, 64, "ReLU", hidden_size=3),
            FlogoResidualBlock(64, 128, "ReLU", hidden_size=1, downsample=torch.nn.Sequential(
                torch.nn.Conv2d(64, 128, (1, 1), (2, 2)),
                torch.nn.BatchNorm2d(128))),
            FlogoResidualBlock(128, 128, "ReLU", hidden_size=3),
            FlogoResidualBlock(128, 256, "ReLU", hidden_size=1, downsample=torch.nn.Sequential(
                torch.nn.Conv2d(128, 256, (1, 1), (2, 2)),
                torch.nn.BatchNorm2d(256))),
            FlogoResidualBlock(256, 256, "ReLU", hidden_size=5),
            FlogoResidualBlock(256, 512, "ReLU", hidden_size=1, downsample=torch.nn.Sequential(
                torch.nn.Conv2d(256, 512, (1, 1), (2, 2)),
                torch.nn.BatchNorm2d(512))),
            FlogoResidualBlock(512, 512, "ReLU", hidden_size=3)]

convolutional2 = [FlogoConvolutionalBlock([Pool(kernel=7, stride=1, padding=0, pool_type="Avg")])]

linear = [FlogoLinearBlock([Linear(512, 10)])]

residualSection = ResidualSection(residual).build()
convolutional1Section = ConvolutionalSection(convolutional1).build()
convolutional2Section = ConvolutionalSection(convolutional2).build()
linearSection = FeedForwardSection(linear).build()

print(convolutional1Section + residualSection + convolutional2Section + linearSection)

# print(torchvision.models.resnet18())

# ForwardTraining(FlogoTraining(10, net, train_data_loader, test_data_loader, FlogoLossFunction("MSELoss"),
#                               FlogoOptimizer("Adam", [i for i in residualModule.parameters()] +
#                                              [i for i in feedForwardModule.parameters()], 0.01))).train()

# RecurrentTrain(FlogoTraining(2, lstmmodel, train_data_loader, test_data_loader,
#                              model.flogo.training.training.FlogoLossFunction("MSELoss"),
#                              model.flogo.training.training.FlogoOptimizer("Adam",
#                                                                           lstmmodel.parameters(), 0.01))).train()
