import os

from flogo.data.columns.categorical import CategoricalColumn
from flogo.data.columns.number import NumericColumn
from flogo.data.dataset_builder import DatasetBuilder
from flogo.data.dataset_splitter import DatasetSplitter
from flogo.data.readers.delimeted_file_reader import DelimitedFileReader
from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer
from flogo.discovery.monitor.accuracy.numeric_monitor import NumericMonitor
from flogo.discovery.regularization.early_stopping import EarlyStopping
from flogo.discovery.regularization.monitors.precision_monitor import PrecisionMonitor
from flogo.discovery.test_task import TestTask
from flogo.discovery.training_task import TrainingTask
from flogo.preprocessing.delete_column import DeleteOperator
from flogo.preprocessing.mappers.leaf.standarization_mapper import StandardizationMapper
from flogo.preprocessing.orchestrator import Orchestrator
from flogo.structure.blocks.flatten import FlattenBlock
from flogo.structure.blocks.linear import LinearBlock
from flogo.structure.blocks.recurrent import RecurrentBlock
from flogo.structure.layers.activation import Activation
from flogo.structure.layers.flatten import Flatten
from flogo.structure.layers.linear import Linear
from flogo.structure.sections.link.flatten import FlattenSection
from flogo.structure.sections.processing.feed_forward import LinearSection
from flogo.structure.sections.processing.recurrent import RecurrentSection
from flogo.structure.structure_factory import StructureFactory
from pytorch.architecture.forward import ForwardArchitecture
from pytorch.discovery.hyperparameters.loss import PytorchLoss
from pytorch.discovery.hyperparameters.optimizer import PytorchOptimizer
from pytorch.discovery.tester import PytorchTester
from pytorch.discovery.trainers.forward_trainer import ForwardTrainer
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

recurrentSection = RecurrentSection([RecurrentBlock(28, 100, 2, "RNN")])

linearSection = LinearSection([LinearBlock([
    Linear(100 * 28, 1000),
    Linear(1000, 250),
    Linear(250, 1)])])

structure = StructureFactory([recurrentSection, linearSection],
                             PytorchGenerator()).create_structure()

architecture = ForwardArchitecture(structure)

model = TrainingTask(epochs, architecture, train_dataset, validation_dataset, Loss(PytorchLoss("MSELoss")),
             Optimizer(PytorchOptimizer("Adam", architecture.parameters(), 0.01)), ForwardTrainer).execute()

# TestTask(test_dataset, PytorchTestTask).test(model)
