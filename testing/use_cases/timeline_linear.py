import os

from flogo.data.dataset.dataset_builder import DatasetBuilder
from flogo.data.dataset.dataset_splitter import DatasetSplitter
from flogo.data.timeline.parser import Parser
from flogo.data.timeline.readers.timeline_reader import TimelineReader
from flogo.data.timeline.utils.metrics import MONTH, DAY, HOUR
from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer
from flogo.discovery.regularization.early_stopping import EarlyStopping
from flogo.discovery.regularization.monitors.precision_monitor import PrecisionMonitor
from flogo.discovery.tasks.test_task import TestTask
from flogo.discovery.tasks.training_task import TrainingTask
from flogo.preprocessing.delete_column import DeleteOperator
from flogo.preprocessing.mappers.leaf.standarization_mapper import StandardizationMapper
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
from pytorch.discovery.measurers.loss_measurer import LossMeasurer
from pytorch.discovery.tester import PytorchTester
from pytorch.discovery.trainer import PytorchTrainer
from pytorch.discovery.validator import PytorchValidator
from pytorch.preprocessing.pytorch_caster import PytorchCaster
from pytorch.structure.generator import PytorchGenerator


def abs_path(part_path):
    return os.path.dirname(os.path.abspath(os.getcwd())) + part_path


epochs = 100

timeline = TimelineReader(Parser()).read(abs_path("/resources/kraken.its"))
dataframe = timeline.group_by(1, HOUR).to_dataframe(20)

dataframe = StandardizationMapper().map(dataframe, ["price_input_0", "price_input_1", "price_input_2", "price_input_3",
                                                    "price_input_4", "price_input_5", "price_input_6", "price_input_7",
                                                    "price_input_8", "price_input_9", "price_input_10",
                                                    "price_input_11",
                                                    "price_input_12", "price_input_13", "price_input_14",
                                                    "price_input_15",
                                                    "price_input_16", "price_input_17", "price_input_18",
                                                    "price_input_19",
                                                    "price_output"])

dataframe = DeleteOperator().delete(dataframe, ["price_input_0", "price_input_1", "price_input_2", "price_input_3",
                                                "price_input_4", "price_input_5", "price_input_6", "price_input_7",
                                                "price_input_8", "price_input_9", "price_input_10", "price_input_11",
                                                "price_input_12", "price_input_13", "price_input_14", "price_input_15",
                                                "price_input_16", "price_input_17", "price_input_18", "price_input_19",
                                                "price_output"])

dataset = DatasetBuilder(PytorchCaster()).build(dataframe,
                                                ["price_input_0'", "price_input_1'", "price_input_2'", "price_input_3'",
                                                 "price_input_4'", "price_input_5'", "price_input_6'", "price_input_7'",
                                                 "price_input_8'", "price_input_9'", "price_input_10'",
                                                 "price_input_11'",
                                                 "price_input_12'", "price_input_13'", "price_input_14'",
                                                 "price_input_15'",
                                                 "price_input_16'", "price_input_17'", "price_input_18'",
                                                 "price_input_19'"], ["price_output'"], 1)

train_dataset, test_dataset, validation_dataset = DatasetSplitter().split(dataset)

linearSection = LinearSection([LinearBlock([
    Linear(20, 100),
    Activation("ReLU"),
    Linear(100, 50),
    Activation("ReLU"),
    Linear(50, 1)])])

structure = StructureFactory([linearSection],
                             PytorchGenerator()).create_structure()

architecture = ForwardArchitecture(structure)

model = TrainingTask(PytorchTrainer(Optimizer(PytorchOptimizer("SGD", architecture.parameters(), 0.01)),
                                    Loss(PytorchLoss("MSELoss"))), PytorchValidator(LossMeasurer()),
                     EarlyStopping(PrecisionMonitor(0))) \
    .execute(epochs, architecture, train_dataset, validation_dataset)

print("Test: ", TestTask(test_dataset, LossMeasurer(), PytorchTester).execute(model))
