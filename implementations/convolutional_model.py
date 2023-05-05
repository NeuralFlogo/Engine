from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer
from flogo.discovery.test_task import TestTask
from flogo.discovery.training_task import TrainingTask
from flogo.structure.blocks.convolutional import ConvolutionalBlock
from flogo.structure.blocks.flatten import FlattenBlock
from flogo.structure.blocks.linear import LinearBlock
from flogo.structure.layers.activation import Activation
from flogo.structure.layers.convolutional import Convolutional
from flogo.structure.layers.flatten import Flatten
from flogo.structure.layers.linear import Linear
from flogo.structure.layers.pool import Pool
from flogo.structure.sections.link.flatten import FlattenSection
from flogo.structure.sections.processing.convolutional import ConvolutionalSection
from flogo.structure.sections.processing.feed_forward import LinearSection
from flogo.structure.structure_factory import StructureFactory
from pytorch.architecture.forward import ForwardArchitecture
from pytorch.discovery.hyperparameters.loss import PytorchLoss
from pytorch.discovery.hyperparameters.optimizer import PytorchOptimizer
from pytorch.discovery.test_task import PytorchTestTask
from pytorch.discovery.trainers.forward_trainer import ForwardTrainer
from pytorch.structure.generator import PytorchGenerator

epochs = 100

parameters = {
    "shuffle": True,
    "size": 50,
    "mean": 0,
    "std": 1,
    "batch_size": 2
}

path = "/Users/jose_juan/Desktop/mnist"

dataset = read_from_dvc(path, "images", PytorchMapper(),  parameters)
train_dataset, test_dataset, validation_dataset = dataset.divide_to(0.2, 0.2)
convolutionalSection = ConvolutionalSection([ConvolutionalBlock([
    Convolutional(1, 6, kernel=5),
    Pool("Max"),
    Activation("ReLU"),
    Convolutional(6, 16, kernel=4),
    Pool("Max"),
    Activation("ReLU")])])

flattenSection = FlattenSection(FlattenBlock(Flatten(1, 3)))

linearSection = LinearSection([LinearBlock([
    Linear(1600, 120),
    Activation("ReLU"),
    Linear(120, 10)])])

# PipelineFactory.create(Convolutional)
#     .dataAdquisition(dataset)
#     .dataPreprocessing(...)
#     .modeling(Structure.of(convolutionalSection)
#     .train(linearSection)
#     .

structure = StructureFactory([convolutionalSection, flattenSection, linearSection],
                             PytorchGenerator()).create_structure()

model = ForwardArchitecture(structure)

print(model)

TrainingTask(epochs, model, train_dataset, validation_dataset, Loss(PytorchLoss("MSELoss")),
             Optimizer(PytorchOptimizer("SGD", model.parameters(), 0.001)), ForwardTrainer).execute()

TestTask(model, test_dataset, PytorchTestTask).test()
