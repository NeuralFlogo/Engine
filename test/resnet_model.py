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

train_loader = torch.utils.data.DataLoader(datasets.CIFAR10("./cifar",
                                                            download=True,
                                                            transform=transforms.Compose([
                                                                transforms.ToTensor(),
                                                                transforms.Resize(size=(224, 224)),
                                                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])),
                                           batch_size=10,
                                           shuffle=True)

# download and transform test dataset
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10("./cifar",
                                                           download=True,
                                                           transform=transforms.Compose([
                                                               transforms.ToTensor(),
                                                               transforms.Resize(size=(224, 224)),
                                                               transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                                    std=[0.2023, 0.1994, 0.2010])])),
                                          batch_size=10,
                                          shuffle=True)

EPOCHS = 200
parameters = ["one-hot"] * 22
train_data_loader, test_data_loader = images_source_type(256, 0, 1, "/Users/jose_juan/Desktop/training", 1)

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

linear = [FlogoLinearBlock([Linear(1280000, 10)])]

residualSection = ResidualSection(residual).build()
convolutional1Section = ConvolutionalSection(convolutional1).build()
convolutional2Section = ConvolutionalSection(convolutional2).build()
flatmapSection = FlattenSection(classification).build()
linearSection = FeedForwardSection(linear).build()

model = ForwardModule(convolutional1Section + residualSection + convolutional2Section + flatmapSection + linearSection)
ForwardTraining(FlogoTraining(EPOCHS, model, training_loader=train_loader, validation_loader=train_loader,
                              loss_function=FlogoLossFunction("CrossEntropyLoss"),
                              optimizer=FlogoOptimizer("SGD", model_params=model.parameters(), lr=0.01))).train()
Testing(model, test_loader).test()
