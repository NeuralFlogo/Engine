import os

from framework.data.dataframe.columns.categorical import CategoricalColumn
from framework.data.dataframe.columns.number import NumericColumn
from framework.data.dataframe.readers.delimeted_file_reader import DelimitedFileReader
from framework.data.dataset.dataset_builder import DatasetBuilder
from framework.data.dataset.dataset_splitter import DatasetSplitter
from framework.discovery.hyperparameters.loss import Loss
from framework.discovery.hyperparameters.optimizer import Optimizer
from framework.discovery.tasks.test_task import TestTask
from framework.discovery.tasks.training_task import TrainingTask
from framework.preprocessing.delete_column import DeleteOperator
from framework.preprocessing.mappers.leaf.one_hot_mapper import OneHotMapper
from framework.preprocessing.mappers.leaf.standarization_mapper import StandardizationMapper
from framework.preprocessing.orchestrator import Orchestrator
from framework.structure.blocks.linear import LinearBlock
from framework.structure.layers.activation import Activation
from framework.structure.layers.linear import Linear
from framework.structure.sections.processing.feed_forward import LinearSection
from framework.structure.structure_factory import StructureFactory
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

dataframe = DelimitedFileReader(",").read(abs_path("/resources/students_performance_dataset.csv"), columns)

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
    Linear(25, 1)
])])

structure = StructureFactory([linearSection], PytorchGenerator()).create_structure()

architecture = ForwardArchitecture(structure)

model = TrainingTask(PytorchTrainer(Optimizer(PytorchOptimizer("Adam", architecture.parameters(), 0.01)),
                                    Loss(PytorchLoss("MSELoss"))), PytorchValidator(LossMeasurer()))\
    .execute(epochs, architecture, train_dataset, validation_dataset)

print("Test: ", TestTask(PytorchTester(test_dataset, LossMeasurer())).execute(model))
