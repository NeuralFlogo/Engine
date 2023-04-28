import torch.nn

from flogo.discovery.test_task import TestTask
from flogo.structure.blocks.convolutional import ConvolutionalBlock
from flogo.structure.blocks.flatten import FlattenBlock
from flogo.structure.blocks.linear import LinearBlock
from flogo.structure.blocks.residual import ResidualBlock
from flogo.structure.layers.activation import Activation
from flogo.structure.layers.convolutional import Convolutional
from flogo.structure.layers.flatten import Flatten
from flogo.structure.layers.linear import Linear
from flogo.structure.layers.normalization import Normalization
from flogo.structure.layers.pool import Pool
from flogo.structure.sections.link.flatten import FlattenSection
from flogo.structure.sections.processing.convolutional import ConvolutionalSection
from flogo.structure.sections.processing.residual import ResidualSection
from pytorch.architecture.forward import ForwardArchitecture

from pytorch.preprocesing.SourceTypeFunctions import images_source_type
from pytorch.structure.sections.processing.feed_forward import FeedForwardSection

EPOCHS = 10
train_loader, test_loader = images_source_type(256, 0, 1,
                                               "/Users/jose_juan/Desktop/monentia/training_set_test", 5)

convolutional1 = [ConvolutionalBlock([Convolutional(3, 64, kernel=7, stride=2, padding=3), Normalization(64),
                                      Activation("ReLU"), Pool(kernel=3, stride=2, padding=1, pool_type="Max")])]

residual = [ResidualBlock(64, 64, "ReLU", hidden_size=3),
            ResidualBlock(64, 128, "ReLU", hidden_size=1, downsample=torch.nn.Sequential(
                torch.nn.Conv2d(64, 128, (1, 1), (1, 1)),
                torch.nn.BatchNorm2d(128))),
            ResidualBlock(128, 128, "ReLU", hidden_size=3),
            ResidualBlock(128, 256, "ReLU", hidden_size=1, downsample=torch.nn.Sequential(
                torch.nn.Conv2d(128, 256, (1, 1), (1, 1)),
                torch.nn.BatchNorm2d(256))),
            ResidualBlock(256, 256, "ReLU", hidden_size=5),
            ResidualBlock(256, 512, "ReLU", hidden_size=1, downsample=torch.nn.Sequential(
                torch.nn.Conv2d(256, 512, (1, 1), (1, 1)),
                torch.nn.BatchNorm2d(512))),
            ResidualBlock(512, 512, "ReLU", hidden_size=3)]

convolutional2 = [ConvolutionalBlock([Pool(kernel=7, stride=1, padding=0, pool_type="Avg")])]

classification = FlattenBlock(Flatten(1, 3))

linear = [LinearBlock([Linear(1722368, 2)])]

residualSection = ResidualSection(residual).__build()
convolutional1Section = ConvolutionalSection(convolutional1).__build()
convolutional2Section = ConvolutionalSection(convolutional2).__build()
flatmapSection = FlattenSection(classification).__build()
linearSection = FeedForwardSection(linear).__build()

# architectures = ForwardModule(convolutional1Section + residualSection + convolutional2Section + flatmapSection +
# linearSection) ForwardTraining(FlogoTraining(EPOCHS, architectures, training_loader=train_loader,
# validation_loader=train_loader, loss_function=FlogoLossFunction("CrossEntropyLoss"), optimizer=FlogoOptimizer(
# "SGD", model_params=architectures.parameters(), lr=0.01))).train()

model = ForwardArchitecture(convolutional1Section + residualSection + convolutional2Section + flatmapSection + linearSection)
model.load_state_dict(torch.load("/Users/jose_juan/PycharmProjects/Flogo/implementations/models/model_20230426_085928_3"))
model.eval()
TestTask(model, test_loader).execute()
