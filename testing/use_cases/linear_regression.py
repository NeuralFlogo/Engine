import os

from flogo.data.dataframe.columns.number import NumericColumn
from flogo.data.dataframe.readers.delimeted_file_reader import DelimitedFileReader
from flogo.data.dataset.dataset_builder import DatasetBuilder
from flogo.data.dataset.dataset_splitter import DatasetSplitter
from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer
from flogo.discovery.tasks.test_task import TestTask

from flogo.discovery.tasks.training_task import TrainingTask
from flogo.structure.blocks.linear import LinearBlock
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


epochs = 100
columns = {"input": NumericColumn(), "output": NumericColumn()}

dataframe = DelimitedFileReader(",").read(abs_path("/testing/resources/regression_dataset.csv"), columns)

dataset = DatasetBuilder(PytorchCaster()).build(dataframe, ["input"], ["output"], 1)

train_dataset, test_dataset, validation_dataset = DatasetSplitter().split(dataset)

linearSection = LinearSection([LinearBlock([
    Linear(1, 1),
])])


training_dataset, test_dataset, validation_dataset = DatasetSplitter().split(dataset, shuffle=False)

structure = StructureFactory([linearSection], PytorchGenerator()).create_structure()

architecture = ForwardArchitecture(structure)

model = TrainingTask(PytorchTrainer(Optimizer(PytorchOptimizer("SGD", architecture.parameters(), 0.01)),
                                    Loss(PytorchLoss("MSELoss"))), PytorchValidator(LossMeasurer()))\
    .execute(epochs, architecture, train_dataset, validation_dataset)

print("Test: ", TestTask(test_dataset, LossMeasurer(), PytorchTester).execute(model))


