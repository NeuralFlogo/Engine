import os

from flogo.data.dataframe.columns.categorical import CategoricalColumn
from flogo.data.dataframe.columns.number import NumericColumn
from flogo.data.dataframe.readers.delimeted_file_reader import DelimitedFileReader
from flogo.data.dataset.dataset_builder import DatasetBuilder
from flogo.data.dataset.dataset_splitter import DatasetSplitter
from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer

from flogo.discovery.regularization.early_stopping import EarlyStopping
from flogo.discovery.regularization.monitors.precision_monitor import PrecisionMonitor
from flogo.discovery.tasks.test_task import TestTask
from flogo.discovery.tasks.training_task import TrainingTask
from flogo.preprocessing.delete_column import DeleteOperator
from flogo.preprocessing.mappers.leaf.one_hot_mapper import OneHotMapper
from flogo.preprocessing.mappers.leaf.standarization_mapper import StandardizationMapper
from flogo.preprocessing.orchestrator import Orchestrator
from flogo.structure.blocks.linear import LinearBlock
from flogo.structure.layers.activation import Activation
from flogo.structure.layers.linear import Linear
from flogo.structure.sections.processing.feed_forward import LinearSection
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


epochs = 200

columns = {"Gender": CategoricalColumn(), "Ethnicity": CategoricalColumn(), "ParentLevel": CategoricalColumn(),
           "Lunch": CategoricalColumn(), "Test": CategoricalColumn(),
           "Math": NumericColumn(dtype=int), "Reading": NumericColumn(dtype=int),
           "Writing": NumericColumn(dtype=int)}

dataframe = DelimitedFileReader(",").read(abs_path("/testing/resources/students_performance_dataset.csv"), columns)

dataframe = DeleteOperator().delete(dataframe, ["Test", "Lunch", "Ethnicity"])


dataframe = Orchestrator(OneHotMapper(), StandardizationMapper()).process(dataframe,
                                                                          ["Gender", "ParentLevel"],
                                                                          ["Math", "Reading", "Writing"])

dataset = DatasetBuilder(PytorchCaster()).build(dataframe, ["Gender_female", "Gender_male",
                                                            "ParentLevel_associate's degree", "Writing'", "Math'"],
                                                ["Reading'"], 3)
train_dataset, test_dataset, validation_dataset = DatasetSplitter().split(dataset)

linearSection = LinearSection([LinearBlock([
    Linear(5, 25),
    Activation("ReLU"),
    Linear(25, 50),
    Activation("ReLU"),
    Linear(50, 25),
    Activation("ReLU"),
    Linear(25, 5)
])])

structure = StructureFactory([linearSection], PytorchGenerator()).create_structure()

architecture = ForwardArchitecture(structure)

model = TrainingTask(PytorchTrainer(Optimizer(PytorchOptimizer("Adam", architecture.parameters(), 0.01)),
                                    Loss(PytorchLoss("MSELoss"))), PytorchValidator(LossMeasurer()),
                     EarlyStopping(PrecisionMonitor(100)))\
    .execute(epochs, architecture, train_dataset, validation_dataset)

# print("Test: ", TestTask(test_dataset, LossMeasurer(), PytorchTester).execute(model))

# for entry in test_dataset:
#     input, output = entry.get_input(), entry.get_output()
#     print("Esperado: ", output)
#     print("Resultado: ", model(input))
# TestTask(test_dataset, PytorchTestTask).test(model)
