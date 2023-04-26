import torch.nn
from torchvision import datasets
from torchvision import transforms

from pytorch.model.sections.link.flatten import FlattenSection
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
from model.flogo.training.loss import FlogoLossFunction
from model.flogo.training.optimizer import FlogoOptimizer
from model.flogo.training.training import FlogoTraining
from pytorch.model.models.forward import ForwardModule
from pytorch.model.sections.processing.convolutional import ConvolutionalSection
from pytorch.model.sections.processing.feed_forward import FeedForwardSection
from pytorch.model.sections.processing.residual import ResidualSection
from pytorch.preprocesing.SourceTypeFunctions import images_source_type
from pytorch.training.forward_train import ForwardTraining
from pytorch.training.test import Testing

EPOCHS = 10
train_loader, test_loader = images_source_type(256, 0, 1,
                                               "/Users/jose_juan/Desktop/monentia/training_set_test", 5)

convolutional1 = [FlogoConvolutionalBlock([Conv(3, 64, kernel=7, stride=2, padding=3), Normalization(64),
                                           Activation("ReLU"), Pool(kernel=3, stride=2, padding=1, pool_type="Max")])]

residual = [FlogoResidualBlock(64, 64, "ReLU", hidden_size=3),
            FlogoResidualBlock(64, 128, "ReLU", hidden_size=1, downsample=torch.nn.Sequential(
                torch.nn.Conv2d(64, 128, (1, 1), (1, 1)),
                torch.nn.BatchNorm2d(128))),
            FlogoResidualBlock(128, 128, "ReLU", hidden_size=3),
            FlogoResidualBlock(128, 256, "ReLU", hidden_size=1, downsample=torch.nn.Sequential(
                torch.nn.Conv2d(128, 256, (1, 1), (1, 1)),
                torch.nn.BatchNorm2d(256))),
            FlogoResidualBlock(256, 256, "ReLU", hidden_size=5),
            FlogoResidualBlock(256, 512, "ReLU", hidden_size=1, downsample=torch.nn.Sequential(
                torch.nn.Conv2d(256, 512, (1, 1), (1, 1)),
                torch.nn.BatchNorm2d(512))),
            FlogoResidualBlock(512, 512, "ReLU", hidden_size=3)]

convolutional2 = [FlogoConvolutionalBlock([Pool(kernel=7, stride=1, padding=0, pool_type="Avg")])]

classification = FlogoFlattenBlock(Flatten(1, 3))

linear = [FlogoLinearBlock([Linear(1722368, 2)])]

residualSection = ResidualSection(residual).build()
convolutional1Section = ConvolutionalSection(convolutional1).build()
convolutional2Section = ConvolutionalSection(convolutional2).build()
flatmapSection = FlattenSection(classification).build()
linearSection = FeedForwardSection(linear).build()

# model = ForwardModule(convolutional1Section + residualSection + convolutional2Section + flatmapSection + linearSection)
# ForwardTraining(FlogoTraining(EPOCHS, model, training_loader=train_loader, validation_loader=train_loader,
#                               loss_function=FlogoLossFunction("CrossEntropyLoss"),
#                               optimizer=FlogoOptimizer("SGD", model_params=model.parameters(), lr=0.01))).train()

model = ForwardModule(convolutional1Section + residualSection + convolutional2Section + flatmapSection + linearSection)
model.load_state_dict(torch.load("/Users/jose_juan/PycharmProjects/Flogo/test/models/model_20230426_085928_3"))
model.eval()
Testing(model, test_loader).test()
