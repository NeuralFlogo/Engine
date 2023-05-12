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
from flogo.discovery.training_task import TrainingTask
from flogo.preprocessing.delete_column import DeleteOperator
from flogo.preprocessing.mappers.leaf.normalization_mapper import NormalizationMapper
from flogo.preprocessing.mappers.leaf.standarization_mapper import StandardizationMapper
from flogo.preprocessing.orchestrator import Orchestrator
from flogo.structure.blocks.linear import LinearBlock
from flogo.structure.blocks.recurrent import RecurrentBlock
from flogo.structure.layers.activation import Activation
from flogo.structure.layers.linear import Linear
from flogo.structure.sections.processing.feed_forward import LinearSection
from flogo.structure.sections.processing.recurrent import RecurrentSection
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

columns = {"Date": CategoricalColumn(), "Temp": NumericColumn(dtype=float),
           "Humidity": NumericColumn(dtype=float), "Wind_speed": NumericColumn(dtype=float),
           "Pressure": NumericColumn(dtype=float)}

dataframe = DelimitedFileReader(",").read(abs_path("/resources/climate_dataset.csv"), columns)

dataframe = DeleteOperator().delete(dataframe, ["Date"])

dataframe = Orchestrator(NormalizationMapper(5, -5)).process(dataframe, ["Temp", "Humidity", "Wind_speed", "Pressure"])

dataset = DatasetBuilder(PytorchCaster()).build(dataframe, ["Temp'", "Humidity'", "Wind_speed'"],["Pressure'"], 1)
train_dataset, test_dataset, validation_dataset = DatasetSplitter().split(dataset, shuffle=False)

recurrentSection = RecurrentSection([RecurrentBlock(3, 100, 2, "RNN")])

linearSection = LinearSection([LinearBlock([
    Linear(100, 50),
    Activation("ReLU"),
    Linear(50, 25),
    Activation("ReLU"),
    Linear(25, 1)])])

structure = StructureFactory([recurrentSection, linearSection],
                             PytorchGenerator()).create_structure()

architecture = ForwardArchitecture(structure)

model = TrainingTask(ForwardTrainer, epochs, architecture, train_dataset, validation_dataset,
                     Loss(PytorchLoss("MSELoss")),
                     Optimizer(PytorchOptimizer("SGD", architecture.parameters(), 0.1)), NumericMonitor(),
                     EarlyStopping(PrecisionMonitor(9000000))).execute()

for entry in test_dataset:
    input, output = entry.get_input(), entry.get_output()
    print("Esperado: ", output)
    print("Resultado: ", model(input))

# TestTask(test_dataset, PytorchTestTask).test(model)
