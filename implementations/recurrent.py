from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer
from flogo.discovery.test_task import TestTask
from flogo.discovery.training_task import TrainingTask
from flogo.preprocesing.datasets.dataset import Dataset
from flogo.preprocesing.readers.file_reader import FileReader
from flogo.preprocesing.transformers.image_transformer import ImageTransformer
from flogo.structure.blocks.linear import LinearBlock
from flogo.structure.blocks.recurrent import RecurrentBlock
from flogo.structure.layers.linear import Linear
from flogo.structure.sections.processing.feed_forward import LinearSection
from flogo.structure.sections.processing.recurrent import RecurrentSection
from flogo.structure.structure_factory import StructureFactory
from pytorch.architecture.forward import ForwardArchitecture
from pytorch.discovery.hyperparameters.loss import PytorchLoss
from pytorch.discovery.hyperparameters.optimizer import PytorchOptimizer
from pytorch.discovery.test_task import PytorchTestTask
from pytorch.discovery.trainers.forward_trainer import ForwardTrainer
from pytorch.preprocessing.mappers.pytorch_mapper import PytorchMapper
from pytorch.preprocessing.preprocessors.images.image_directory_preprocessor import ImageDirectoryPreprocessor
from pytorch.structure.generator import PytorchGenerator

path = "/Users/jose_juan/Desktop/mnist"
epochs = 100

dataset = Dataset.get(ImageTransformer(FileReader(path), ImageDirectoryPreprocessor(28, mean=0, std=1), True), PytorchMapper(), 64)

train_dataset, test_dataset, validation_dataset = dataset.divide_to(0.2, 0.2)

recurrentSection = RecurrentSection([RecurrentBlock(28, 256, 2, "RNN")])

linearSection = LinearSection([LinearBlock([
    Linear(256 * 28, 10)])])

structure = StructureFactory([recurrentSection, linearSection],
                             PytorchGenerator()).create_structure()

architecture = ForwardArchitecture(structure)

TrainingTask(epochs, architecture, train_dataset, validation_dataset, Loss(PytorchLoss("CrossEntropyLoss")),
             Optimizer(PytorchOptimizer("Adam", architecture.parameters(), 0.001)), ForwardTrainer).execute()

TestTask(architecture, test_dataset, PytorchTestTask).test()
