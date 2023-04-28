from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer
from flogo.discovery.training_task import TrainingTask
from flogo.structure.blocks.classification import ClassificationBlock
from flogo.structure.blocks.convolutional import ConvolutionalBlock
from flogo.structure.blocks.flatten import FlattenBlock
from flogo.structure.blocks.linear import LinearBlock
from flogo.structure.layers.activation import Activation
from flogo.structure.layers.classification import Classification
from flogo.structure.layers.convolutional import Convolutional
from flogo.structure.layers.flatten import Flatten
from flogo.structure.layers.linear import Linear
from flogo.structure.layers.pool import Pool
from flogo.structure.sections.link.classificationsection import ClassificationSection
from flogo.structure.sections.link.flatten import FlattenSection
from flogo.structure.sections.processing.convolutional import ConvolutionalSection
from flogo.structure.sections.processing.feed_forward import LinearSection
from flogo.structure.structure_factory import StructureFactory
from pytorch.architecture.forward import ForwardArchitecture
from pytorch.datasets.mappers.pytorch_mapper import PytorchMapper
from pytorch.discovery.hyperparameters.loss import PytorchLoss
from pytorch.discovery.hyperparameters.optimizer import PytorchOptimizer
from pytorch.discovery.trainers.forward_trainer import ForwardTrainer
from pytorch.preprocesing.dvc_utils import read_from_dvc
from pytorch.structure.generator import PytorchGenerator

epochs = 200

parameters = {
    "shuffle": True,
    "size": 50,
    "mean": 0,
    "std": 1,
    "batch_size": 2
}

path = "/Users/jose_juan/Desktop/prueba"

dataset = read_from_dvc(path, "images", PytorchMapper(),  parameters)

train_dataset, test_dataset, validation_dataset = dataset.divide_to(0.2, 0.2)

convolutionalSection = ConvolutionalSection([ConvolutionalBlock([
    Convolutional(3, 50),
    Activation("ReLU"),
    Convolutional(50, 100),
    Activation("ReLU"),
    Pool("Max"),
    Convolutional(100, 250),
    Activation("ReLU"),
    Convolutional(250, 500),
    Activation("ReLU"),
    Pool("Max"),
    Convolutional(500, 250),
    Activation("ReLU"),
    Convolutional(250, 100),
    Activation("ReLU"),
    Pool("Max")])])

flattenSection = FlattenSection(FlattenBlock(Flatten(1, 3)))

linearSection = LinearSection([LinearBlock([
    Linear(400, 150),
    Activation("ReLU"),
    Linear(150, 200),
    Activation("ReLU"),
    Linear(200, 100),
    Activation("ReLU"),
    Linear(100, 50),
    Activation("ReLU"),
    Linear(50, 10),
    Activation("ReLU"),
    Linear(10, 2)])])

classificationSection = ClassificationSection(ClassificationBlock(Classification("Softmax", 1)))

structure = StructureFactory([convolutionalSection, flattenSection, linearSection, classificationSection],
                             PytorchGenerator()).create_structure()

model = ForwardArchitecture(structure)

print(model)

TrainingTask(epochs, model, train_dataset, validation_dataset, Loss(PytorchLoss("MSELoss")),
             Optimizer(PytorchOptimizer("SGD", model.parameters(), 1)), ForwardTrainer).execute()
