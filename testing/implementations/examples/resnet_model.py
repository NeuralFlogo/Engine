import torch.nn

from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer
from flogo.discovery.test_task import TestTask
from flogo.discovery.training_task import TrainingTask
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
from flogo.structure.sections.processing.feed_forward import LinearSection
from flogo.structure.sections.processing.residual import ResidualSection
from flogo.structure.structure_factory import StructureFactory
from pytorch.architecture.forward import ForwardArchitecture
from pytorch.datasets.mappers.PytorchMapper import PytorchMapper
from pytorch.discovery.hyperparameters.loss import PytorchLoss
from pytorch.discovery.hyperparameters.optimizer import PytorchOptimizer
from pytorch.discovery.test_task import PytorchTestTask
from pytorch.discovery.trainers.forward_trainer import ForwardTrainer
from pytorch.preprocesing.SourceTypeFunctions import images_source_type
from pytorch.preprocesing.dvc_utils import read_from_dvc
from pytorch.structure.generator import PytorchGenerator

parameters = {
    "shuffle": True,
    "size": 50,
    "mean": 0,
    "std": 1,
    "batch_size": 2
}

epochs = 10

path = "/Users/jose_juan/Desktop/mnist"

dataset = read_from_dvc(path, "images", PytorchMapper(),  parameters)

train_dataset, test_dataset, validation_dataset = dataset.divide_to(0.2, 0.2)

convolutional1 = ConvolutionalSection([ConvolutionalBlock([Convolutional(1, 64, kernel=7, stride=2, padding=3),
                                                           Normalization(64),
                                                           Activation("ReLU"),
                                                           Pool(kernel=3, stride=2, padding=1, pool_type="Max")])])

residual = ResidualSection([ResidualBlock(64, 64, "ReLU", hidden_size=3),
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
                            ResidualBlock(512, 512, "ReLU", hidden_size=3)])

convolutional2 = ConvolutionalSection([ConvolutionalBlock([Pool(kernel=7, stride=1, padding=0, pool_type="Avg")])])

flatten = FlattenSection(FlattenBlock(Flatten(1, 3)))

linear = LinearSection([LinearBlock([Linear(25088, 10)])])

structure = StructureFactory([convolutional1, residual, convolutional2, flatten, linear],
                             PytorchGenerator()).create_structure()

architecture = ForwardArchitecture(structure)

TrainingTask(epochs, architecture, train_dataset, validation_dataset, Loss(PytorchLoss("MSELoss")),
             Optimizer(PytorchOptimizer("Adam", architecture.parameters(), 0.001)), ForwardTrainer).execute()

TestTask(architecture, test_dataset, PytorchTestTask).test()

