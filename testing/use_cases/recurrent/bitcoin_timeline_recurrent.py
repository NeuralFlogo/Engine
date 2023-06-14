import os

from framework.data.dataset.dataset_builder import DatasetBuilder
from framework.data.dataset.dataset_splitter import DatasetSplitter
from framework.data.timeline.parser import Parser
from framework.data.timeline.readers.timeline_reader import TimelineReader
from framework.data.timeline.utils.metrics import DAY
from framework.discovery.hyperparameters.loss import Loss
from framework.discovery.hyperparameters.optimizer import Optimizer
from framework.discovery.regularization.early_stopping import EarlyStopping
from framework.discovery.regularization.monitors.precision_monitor import PrecisionMonitor
from framework.discovery.tasks.test_task import TestTask
from framework.discovery.tasks.training_task import TrainingTask
from framework.preprocessing.delete_column import DeleteOperator
from framework.preprocessing.mappers.leaf.standarization_mapper import StandardizationMapper
from framework.structure.blocks.linear import LinearBlock
from framework.structure.blocks.recurrent import RecurrentBlock
from framework.structure.layers.activation import Activation
from framework.structure.layers.linear import Linear
from framework.structure.sections.processing.linear import LinearSection
from framework.structure.sections.processing.recurrent import RecurrentSection
from framework.structure.structure_launcher import StructureLauncher
from pytorch.architecture.forward import ForwardArchitecture
from pytorch.data.torch_gpu_entry_allocator import TorchGpuEntryAllocator
from pytorch.discovery.hyperparameters.loss import PytorchLoss
from pytorch.discovery.hyperparameters.optimizer import PytorchOptimizer
from pytorch.discovery.measurers.loss_measurer import LossMeasurer
from pytorch.discovery.tester import PytorchTester
from pytorch.discovery.trainer import PytorchTrainer
from pytorch.discovery.validator import PytorchValidator
from pytorch.preprocessing.pytorch_caster import PytorchCaster
from pytorch.structure.interpreter import PytorchInterpreter


def abs_path(part_path):
    return os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))) + part_path


epochs = 100

timeline = TimelineReader(Parser()).read(abs_path("/resources/kraken.its"))
dataframe = timeline.group_by(2, DAY).to_dataframe(2)

dataframe = StandardizationMapper().map(dataframe, ["price_input_0", "price_input_1", "price_output"])

dataframe = DeleteOperator().delete(dataframe, ["price_input_0", "price_input_1", "price_output"])

dataset = DatasetBuilder(PytorchCaster()).build(dataframe,
                                                ["price_input_0'", "price_input_1'"], ["price_output'"], 5)
train_dataset, test_dataset, validation_dataset = DatasetSplitter().split(dataset)

recurrentSection = RecurrentSection([RecurrentBlock(2, 200, 2, "RNN")])

linearSection = LinearSection([LinearBlock([
    Linear(200, 100),
    Activation("ReLU"),
    Linear(100, 50),
    Activation("ReLU"),
    Linear(50, 1)])])

structure = StructureLauncher([recurrentSection, linearSection],
                              PytorchInterpreter()).launch()

architecture = ForwardArchitecture(structure)
architecture.to_device("cuda")

model = TrainingTask(PytorchTrainer(Optimizer(PytorchOptimizer("SGD", architecture.parameters(), 0.01)),
                                    Loss(PytorchLoss("MSELoss")),
                                    TorchGpuEntryAllocator()),
                     PytorchValidator(LossMeasurer(), TorchGpuEntryAllocator()),
                     EarlyStopping(PrecisionMonitor(0))) \
    .execute(epochs, architecture, train_dataset, validation_dataset)

print("Test: ", TestTask(PytorchTester(test_dataset, LossMeasurer(), TorchGpuEntryAllocator())).execute(model))
