import os

from flogo.data.columns.number import NumericColumn
from flogo.data.dataset_builder import DatasetBuilder
from flogo.data.dataset_splitter import DatasetSplitter
from flogo.data.readers.delimeted_file_reader import DelimitedFileReader
from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer
from flogo.discovery.monitor.accuracy.numeric_monitor import NumericMonitor
from flogo.discovery.regularization.early_stopping import EarlyStopping
from flogo.discovery.regularization.monitors.precision_monitor import PrecisionMonitor
from flogo.discovery.training_task import TrainingTask
from flogo.structure.blocks.linear import LinearBlock
from flogo.structure.layers.linear import Linear
from flogo.structure.sections.processing.feed_forward import LinearSection
from flogo.structure.structure_factory import StructureFactory
from pytorch.architecture.forward import ForwardArchitecture
from pytorch.discovery.hyperparameters.loss import PytorchLoss
from pytorch.discovery.hyperparameters.optimizer import PytorchOptimizer
from pytorch.discovery.trainers.forward_trainer import ForwardTrainer
from pytorch.preprocessing.pytorch_caster import PytorchCaster
from pytorch.structure.generator import PytorchGenerator


def abs_path(part_path):
    return os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))) + part_path


epochs = 100
columns = {"input": NumericColumn(), "output": NumericColumn()}

dataframe = DelimitedFileReader(",").read(abs_path("/resources/regression_dataset.csv"), columns)

linearSection = LinearSection([LinearBlock([
    Linear(1, 1),
])])

dataset = DatasetBuilder(PytorchCaster()).build(dataframe, ["input"], ["output"], 1)

training_dataset, test_dataset, validation_dataset = DatasetSplitter().split(dataset, 0, 0, shuffle=True)

structure = StructureFactory([linearSection], PytorchGenerator()).create_structure()

architecture = ForwardArchitecture(structure)

model = TrainingTask(ForwardTrainer, epochs, architecture, training_dataset, training_dataset,
                     Loss(PytorchLoss("MSELoss")),
                     Optimizer(PytorchOptimizer("SGD", architecture.parameters(), 0.001)), NumericMonitor(),
                     EarlyStopping(PrecisionMonitor(9000000))).execute()

