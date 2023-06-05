import os

import torch.nn

from framework.data.dataframe.columns.loaded_image import LoadedImageColumn
from framework.data.dataframe.readers.image_reader import ImageReader
from framework.data.dataset.dataset_builder import DatasetBuilder
from framework.data.dataset.dataset_splitter import DatasetSplitter
from framework.discovery.hyperparameters.loss import Loss
from framework.discovery.hyperparameters.optimizer import Optimizer
from framework.discovery.regularization.early_stopping import EarlyStopping
from framework.discovery.regularization.monitors.growth_monitor import GrowthMonitor
from framework.discovery.tasks.test_task import TestTask
from framework.discovery.tasks.training_task import TrainingTask
from framework.preprocessing.mappers.composite import CompositeMapper
from framework.preprocessing.mappers.leaf.one_hot_mapper import OneHotMapper
from framework.preprocessing.mappers.leaf.resize_mapper import ResizeMapper
from framework.preprocessing.mappers.leaf.type_mapper import TypeMapper
from framework.preprocessing.orchestrator import Orchestrator
from framework.structure.blocks.convolutional import ConvolutionalBlock
from framework.structure.blocks.flatten import FlattenBlock
from framework.structure.blocks.linear import LinearBlock
from framework.structure.blocks.residual import ResidualBlock
from framework.structure.layers.activation import Activation
from framework.structure.layers.convolutional import Convolutional
from framework.structure.layers.flatten import Flatten
from framework.structure.layers.linear import Linear
from framework.structure.layers.normalization import Normalization
from framework.structure.layers.pool import Pool
from framework.structure.sections.link.flatten import FlattenSection
from framework.structure.sections.processing.convolutional import ConvolutionalSection
from framework.structure.sections.processing.linear import LinearSection
from framework.structure.sections.processing.residual import ResidualSection
from framework.structure.structure_factory import StructureFactory
from pytorch.architecture.forward import ForwardArchitecture
from pytorch.discovery.hyperparameters.loss import PytorchLoss
from pytorch.discovery.hyperparameters.optimizer import PytorchOptimizer
from pytorch.discovery.measurers.accuracy_measurer import AccuracyMeasurer
from pytorch.discovery.tester import PytorchTester
from pytorch.discovery.trainer import PytorchTrainer
from pytorch.discovery.validator import PytorchValidator
from pytorch.preprocessing.pytorch_caster import PytorchCaster
from pytorch.structure.generator import PytorchGenerator

epochs = 10


def abs_path(part_path):
    return os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))) + part_path


path = abs_path("/resources/mnist")
dataframe = ImageReader().read(path)
dataframe = Orchestrator(OneHotMapper(), CompositeMapper([TypeMapper(LoadedImageColumn), ResizeMapper((50, 50))])) \
    .process(dataframe, ["output"], ["input"])

dataset = DatasetBuilder(PytorchCaster()).build(dataframe, ["input'"], ["output_0", "output_1", "output_2", "output_3",
                                                                        "output_4", "output_5", "output_6", "output_7",
                                                                        "output_8", "output_9"], 5)
train_dataset, test_dataset, validation_dataset = DatasetSplitter().split(dataset)

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

model = TrainingTask(PytorchTrainer(Optimizer(PytorchOptimizer("SGD", architecture.parameters(), 0.01)),
                                    Loss(PytorchLoss("CrossEntropyLoss"))), PytorchValidator(AccuracyMeasurer()),
                     EarlyStopping(GrowthMonitor(1, 0.001))) \
    .execute(epochs, architecture, train_dataset, validation_dataset)

print("Test: ", TestTask(PytorchTester(test_dataset, AccuracyMeasurer())).execute(model))
