import torch.nn

from flogo.data.columns.loaded_image import LoadedImageColumn
from flogo.data.dataset_builder import DatasetBuilder
from flogo.data.dataset_splitter import DatasetSplitter
from flogo.data.readers.image_reader import ImageReader
from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer
from flogo.discovery.regularization.early_stopping import EarlyStopping
from flogo.discovery.regularization.monitors.growth_monitor import GrowthMonitor
from flogo.discovery.tasks.test_task import TestTask
from flogo.discovery.tasks.training_task import TrainingTask
from flogo.preprocessing.mappers.composite import CompositeMapper
from flogo.preprocessing.mappers.leaf.one_hot_mapper import OneHotMapper
from flogo.preprocessing.mappers.leaf.resize_mapper import ResizeMapper
from flogo.preprocessing.mappers.leaf.type_mapper import TypeMapper
from flogo.preprocessing.orchestrator import Orchestrator
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
from pytorch.discovery.hyperparameters.loss import PytorchLoss
from pytorch.discovery.hyperparameters.optimizer import PytorchOptimizer
from pytorch.discovery.measurers.accuracy_measurer import AccuracyMeasurer
from pytorch.discovery.tester import PytorchTester
from pytorch.discovery.trainer import PytorchTrainer
from pytorch.discovery.validator import PytorchValidator
from pytorch.preprocessing.pytorch_caster import PytorchCaster
from pytorch.structure.generator import PytorchGenerator

epochs = 10

path = "C:/Users/Joel/Desktop/mnist"
dataframe = ImageReader().read(path)
dataframe = Orchestrator(OneHotMapper(), CompositeMapper([TypeMapper(LoadedImageColumn), ResizeMapper((50, 50))]))\
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
                     EarlyStopping(GrowthMonitor(1, 0.1)))\
    .execute(epochs, architecture, train_dataset, validation_dataset)

print("Test: ", TestTask(test_dataset, AccuracyMeasurer(), PytorchTester).execute(model))

