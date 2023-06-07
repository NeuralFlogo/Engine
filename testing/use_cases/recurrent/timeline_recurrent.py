import os

from framework.data.dataframe.columns.categorical import CategoricalColumn
from framework.data.dataframe.columns.number import NumericColumn
from framework.data.dataframe.readers.delimeted_file_reader import DelimitedFileReader
from framework.data.dataset.dataset_builder import DatasetBuilder
from framework.data.dataset.dataset_splitter import DatasetSplitter
from framework.discovery.hyperparameters.loss import Loss
from framework.discovery.hyperparameters.optimizer import Optimizer
from framework.discovery.regularization.early_stopping import EarlyStopping
from framework.discovery.regularization.monitors.growth_monitor import GrowthMonitor
from framework.discovery.tasks.test_task import TestTask
from framework.discovery.tasks.training_task import TrainingTask
from framework.preprocessing.delete_column import DeleteOperator
from framework.preprocessing.mappers.leaf.standarization_mapper import StandardizationMapper
from framework.preprocessing.orchestrator import Orchestrator
from framework.structure.blocks.linear import LinearBlock
from framework.structure.blocks.recurrent import RecurrentBlock
from framework.structure.layers.linear import Linear
from framework.structure.sections.processing.linear import LinearSection
from framework.structure.sections.processing.recurrent import RecurrentSection
from framework.structure.structure_factory import StructureFactory
from pytorch.architecture.forward import ForwardArchitecture
from pytorch.data.torch_gpu_entry_allocator import TorchGpuEntryAllocator
from pytorch.discovery.hyperparameters.loss import PytorchLoss
from pytorch.discovery.hyperparameters.optimizer import PytorchOptimizer
from pytorch.discovery.measurers.loss_measurer import LossMeasurer
from pytorch.discovery.tester import PytorchTester
from pytorch.discovery.trainer import PytorchTrainer
from pytorch.discovery.validator import PytorchValidator
from pytorch.preprocessing.pytorch_caster import PytorchCaster
from pytorch.structure.generator import PytorchGenerator


def abs_path(part_path):
    return os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))) + part_path


epochs = 100
columns = {"ts": CategoricalColumn(), "open": NumericColumn(dtype=float), "high": NumericColumn(dtype=float),
           "low": NumericColumn(dtype=float),
           "close": NumericColumn(dtype=float), "volume": NumericColumn(dtype=float)}

dataframe = DelimitedFileReader(",").read(abs_path("/resources/time_series_dataset.csv"), columns)
DeleteOperator().delete(dataframe, ["ts"])

dataframe = Orchestrator(StandardizationMapper()) \
    .process(dataframe, ["open", "high", "low", "volume", "close"])


dataset = DatasetBuilder(PytorchCaster()).build(dataframe, ["open'", "high'", "close'", "volume'"], ["close'"], 1)
train_dataset, test_dataset, validation_dataset = DatasetSplitter().split(dataset)

recurrentSection = RecurrentSection([RecurrentBlock(4, 200, 2, "RNN")])

linearSection = LinearSection([LinearBlock([
    Linear(200, 100),
    Linear(100, 50),
    Linear(50, 1)])])

structure = StructureFactory([recurrentSection, linearSection],
                             PytorchGenerator()).create_structure()

architecture = ForwardArchitecture(structure)
architecture.to_device("cuda")

model = TrainingTask(PytorchTrainer(Optimizer(PytorchOptimizer("SGD", architecture.parameters(), 0.01)),
                                    Loss(PytorchLoss("MSELoss")),
                                    TorchGpuEntryAllocator()),
                     PytorchValidator(LossMeasurer(), TorchGpuEntryAllocator())) \
    .execute(epochs, architecture, train_dataset, validation_dataset)

print("Test: ", TestTask(PytorchTester(test_dataset, LossMeasurer(), TorchGpuEntryAllocator())).execute(model))
