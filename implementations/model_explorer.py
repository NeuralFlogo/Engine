from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer
from flogo.discovery.model_explorer import ModelExplorer
from flogo.discovery.regularization.early_stopping import EarlyStopping
from flogo.discovery.regularization.monitors.loss_monitor import LossMonitor
from flogo.discovery.test_task import TestTask
from flogo.discovery.training_task import TrainingTask
from flogo.preprocesing.datasets.dataset import Dataset
from flogo.preprocesing.readers.file_reader import FileReader
from flogo.preprocesing.transformers.numeric_transformer import NumericTransformer
from flogo.structure.blocks.classification import ClassificationBlock
from flogo.structure.blocks.linear import LinearBlock
from flogo.structure.layers.activation import Activation
from flogo.structure.layers.classification import Classification
from flogo.structure.layers.linear import Linear
from flogo.structure.sections.link.classificationsection import ClassificationSection
from flogo.structure.sections.processing.feed_forward import LinearSection
from flogo.structure.structure_factory import StructureFactory
from pytorch.architecture.forward import ForwardArchitecture
from pytorch.discovery.hyperparameters.loss import PytorchLoss
from pytorch.discovery.hyperparameters.optimizer import PytorchOptimizer
from pytorch.discovery.test_task import PytorchTestTask
from pytorch.discovery.trainers.forward_trainer import ForwardTrainer
from pytorch.preprocessing.mappers.pytorch_mapper import PytorchMapper
from pytorch.preprocessing.preprocessors.numbers.one_hot_preprocessor import OneHotPreprocessor
from pytorch.structure.generator import PytorchGenerator

path = "/Users/jose_juan/Desktop/mushrooms.csv"
preprocessors = [OneHotPreprocessor()] * 22
epochs = 101

dataset = Dataset.get(NumericTransformer(FileReader(path), preprocessors, True), PytorchMapper(), 50)

train_dataset, test_dataset, validation_dataset = dataset.divide_to(0.2, 0.2)

linearSection = LinearSection([LinearBlock([
    Linear(117, 200),
    Activation("ReLU"),
    Linear(200, 150),
    Activation("ReLU"),
    Linear(150, 75),
    Activation("ReLU"),
    Linear(75, 10),
    Activation("ReLU"),
    Linear(10, 2)
])])

classificationSection = ClassificationSection(ClassificationBlock(Classification("Softmax", 1)))

structure = StructureFactory([linearSection, classificationSection], PytorchGenerator()).create_structure()

architecture = ForwardArchitecture(structure)

_, accuracy = ModelExplorer(
    [TrainingTask(epochs, architecture, train_dataset, validation_dataset, Loss(PytorchLoss("MSELoss")),
                  Optimizer(PytorchOptimizer("SGD", architecture.parameters(), 0.1)), ForwardTrainer,
                  early_stopping=EarlyStopping(LossMonitor(5, 0.005))),
     TrainingTask(epochs, architecture, train_dataset, validation_dataset, Loss(PytorchLoss("MSELoss")),
                  Optimizer(PytorchOptimizer("Adam", architecture.parameters(), 0.1)), ForwardTrainer,
                  early_stopping=EarlyStopping(LossMonitor(5, 0.005)))],
    TestTask(test_dataset, PytorchTestTask)).explore()

print(accuracy)
