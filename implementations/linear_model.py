from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer
from flogo.discovery.test_task import TestTask
from flogo.discovery.training_task import TrainingTask
from flogo.structure.blocks.linear import LinearBlock
from flogo.structure.layers.activation import Activation
from flogo.structure.layers.classification import Classification
from flogo.structure.layers.linear import Linear
from flogo.structure.sections.link.classificationsection import ClassificationSection
from flogo.structure.sections.processing.feed_forward import LinearSection
from flogo.structure.structure_factory import StructureFactory
from pytorch.architecture.forward import ForwardArchitecture
from pytorch.datasets.mappers.PytorchMapper import PytorchMapper
from pytorch.discovery.hyperparameters.loss import PytorchLoss
from pytorch.discovery.hyperparameters.optimizer import PytorchOptimizer
from pytorch.discovery.test_task import PytorchTestTask
from pytorch.discovery.trainers.forward_trainer import ForwardTrainer
from pytorch.preprocesing.dvc_utils import read_from_dvc
from pytorch.structure.generator import PytorchGenerator

BATCH_SIZE = 50
epochs = 20
parameters = ["one-hot"] * 22

parameters = {
    "shuffle": True,
    "preprocessing": parameters,
    "batch_size": 50
}

path = "/Users/jose_juan/Desktop/mushrooms.csv"

dataset = read_from_dvc(path, "numeric", PytorchMapper(),  parameters)

train_dataset, test_dataset, validation_dataset = dataset.divide_to(0.2, 0.2)

linearSection = LinearSection([LinearBlock([
    Linear(110, 150),
    Activation("ReLU"),
    Linear(150, 200),
    Activation("ReLU"),
    Linear(200, 100),
    Activation("ReLU"),
    Linear(100, 50),
    Activation("ReLU"),
    Linear(50, 10),
    Activation("ReLU"),
    Linear(10, 2)
])])

classificationSection = ClassificationSection(Classification("Softmax", 1))

structure = StructureFactory([linearSection, classificationSection], PytorchGenerator()).create_structure()

model = ForwardArchitecture(structure)

TrainingTask(epochs, model, train_dataset, validation_dataset, Loss(PytorchLoss("MSELoss")),
             Optimizer(PytorchOptimizer("SGD", model.parameters(), 0.1)), ForwardTrainer).execute()

TestTask(model, test_dataset, PytorchTestTask).test()
