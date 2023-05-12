import os

from flogo.data.columns.categorical import CategoricalColumn
from flogo.data.columns.number import NumericColumn
from flogo.data.dataset_builder import DatasetBuilder
from flogo.data.dataset_splitter import DatasetSplitter
from flogo.data.readers.delimeted_file_reader import DelimitedFileReader
from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer
from flogo.discovery.regularization.early_stopping import EarlyStopping
from flogo.discovery.regularization.monitors.precision_monitor import PrecisionMonitor
from flogo.discovery.tasks.test_task import TestTask
from flogo.discovery.tasks.training_task import TrainingTask
from flogo.preprocessing.delete_column import DeleteOperator
from flogo.preprocessing.mappers.leaf.standarization_mapper import StandardizationMapper
from flogo.preprocessing.orchestrator import Orchestrator
from flogo.structure.blocks.linear import LinearBlock
from flogo.structure.blocks.recurrent import RecurrentBlock
from flogo.structure.layers.linear import Linear
from flogo.structure.sections.processing.feed_forward import LinearSection
from flogo.structure.sections.processing.recurrent import RecurrentSection
from flogo.structure.structure_factory import StructureFactory
from pytorch.architecture.forward import ForwardArchitecture
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

model = TrainingTask(PytorchTrainer(Optimizer(PytorchOptimizer("SGD", architecture.parameters(), 0.01)),
                                    Loss(PytorchLoss("MSELoss"))), PytorchValidator(LossMeasurer()),
                     EarlyStopping(PrecisionMonitor(0)))\
    .execute(epochs, architecture, train_dataset, validation_dataset)

print("Test: ", TestTask(test_dataset, LossMeasurer(), PytorchTester).execute(model))


# TestTask(test_dataset, PytorchTestTask).test(model)
