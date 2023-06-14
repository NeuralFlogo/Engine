import os

from framework.data.dataframe.columns.loaded_image import LoadedImageColumn
from framework.data.dataframe.readers.image_reader import ImageReader
from framework.data.dataset.dataset_builder import DatasetBuilder
from framework.data.dataset.dataset_splitter import DatasetSplitter
from framework.discovery.hyperparameters.loss import Loss
from framework.discovery.hyperparameters.optimizer import Optimizer
from framework.discovery.tasks.test_task import TestTask
from framework.discovery.tasks.training_task import TrainingTask
from framework.preprocessing.mappers.composite import CompositeMapper
from framework.preprocessing.mappers.leaf.one_hot_mapper import OneHotMapper
from framework.preprocessing.mappers.leaf.resize_mapper import ResizeMapper
from framework.preprocessing.mappers.leaf.type_mapper import TypeMapper
from framework.preprocessing.orchestrator import Orchestrator
from framework.structure.blocks.classification import ClassificationBlock
from framework.structure.blocks.convolutional import ConvolutionalBlock
from framework.structure.blocks.flatten import FlattenBlock
from framework.structure.blocks.linear import LinearBlock
from framework.structure.layers.activation import Activation
from framework.structure.layers.classification import Classification
from framework.structure.layers.convolutional import Convolutional
from framework.structure.layers.flatten import Flatten
from framework.structure.layers.linear import Linear
from framework.structure.layers.pool import Pool
from framework.structure.sections.link.classificationsection import ClassificationSection
from framework.structure.sections.link.flatten import FlattenSection
from framework.structure.sections.processing.convolutional import ConvolutionalSection
from framework.structure.sections.processing.linear import LinearSection
from framework.structure.structure_launcher import StructureLauncher
from pytorch.architecture.forward import ForwardArchitecture
from pytorch.architecture.torch_freezer import TorchFreezer
from pytorch.data.torch_gpu_entry_allocator import TorchGpuEntryAllocator
from pytorch.discovery.hyperparameters.loss import PytorchLoss
from pytorch.discovery.hyperparameters.optimizer import PytorchOptimizer
from pytorch.discovery.measurers.accuracy_measurer import AccuracyMeasurer
from pytorch.discovery.tester import PytorchTester
from pytorch.discovery.trainer import PytorchTrainer
from pytorch.discovery.validator import PytorchValidator
from pytorch.preprocessing.pytorch_caster import PytorchCaster
from pytorch.structure.interpreter import PytorchInterpreter


def abs_path(part_path):
    return os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))) + part_path


path = abs_path("/resources/mnist")

epochs = 15

dataframe = ImageReader().read(path)
dataframe = Orchestrator(OneHotMapper(), CompositeMapper([TypeMapper(LoadedImageColumn), ResizeMapper((50, 50))])) \
    .process(dataframe, ["output"], ["input"])

dataset = DatasetBuilder(PytorchCaster()).build(dataframe, ["input'"], ["output_0", "output_1", "output_2", "output_3",
                                                                        "output_4", "output_5", "output_6", "output_7",
                                                                        "output_8", "output_9"], 1)
train_dataset, test_dataset, validation_dataset = DatasetSplitter().split(dataset, shuffle=True)

convolutionalSection = ConvolutionalSection([ConvolutionalBlock([
    Convolutional(1, 6, kernel=5),
    Pool("Max"),
    Activation("ReLU"),
    Convolutional(6, 16, kernel=4),
    Pool("Max"),
    Activation("ReLU")])])

flattenSection = FlattenSection(FlattenBlock(Flatten(1, 3)))

linearSection = LinearSection([LinearBlock([
    Linear(1600, 120),
    Activation("ReLU"),
    Linear(120, 10)])])

classificationSection = ClassificationSection(ClassificationBlock(Classification("Softmax", 1)))

structure = StructureLauncher([convolutionalSection, flattenSection, linearSection, classificationSection],
                              PytorchInterpreter()).launch()

architecture = ForwardArchitecture(structure)
architecture.to_device("cuda")

model = TrainingTask(PytorchTrainer(Optimizer(PytorchOptimizer("SGD", architecture.parameters(), 0.01)),
                                    Loss(PytorchLoss("CrossEntropyLoss")),
                                    TorchGpuEntryAllocator()),
                     PytorchValidator(AccuracyMeasurer(), TorchGpuEntryAllocator())) \
    .execute(epochs, architecture, train_dataset, validation_dataset)

print("Test: ", TestTask(PytorchTester(test_dataset, AccuracyMeasurer(), TorchGpuEntryAllocator())).execute(model))

TorchFreezer().freeze(model.get_section_range(0, 1))

model = TrainingTask(PytorchTrainer(Optimizer(PytorchOptimizer("SGD", architecture.parameters(), 0.01)),
                                    Loss(PytorchLoss("CrossEntropyLoss")),
                                    TorchGpuEntryAllocator()),
                     PytorchValidator(AccuracyMeasurer(), TorchGpuEntryAllocator())) \
    .execute(epochs, architecture, train_dataset, validation_dataset)

print("Test: ", TestTask(PytorchTester(test_dataset, AccuracyMeasurer(), TorchGpuEntryAllocator())).execute(model))
