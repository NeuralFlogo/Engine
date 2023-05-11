from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer
from flogo.discovery.regularization.early_stopping import EarlyStopping
from flogo.discovery.regularization.monitors.loss_monitor import LossMonitor
from flogo.discovery.test_task import TestTask
from flogo.discovery.training_task import TrainingTask
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
from pytorch.structure.generator import PytorchGenerator

epochs = 10

linearSection = LinearSection([LinearBlock([
    Linear(117, 150),
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

classificationSection = ClassificationSection(ClassificationBlock(Classification("Softmax", 1)))

structure = StructureFactory([linearSection, classificationSection], PytorchGenerator()).create_structure()

architecture = ForwardArchitecture(structure)

model = TrainingTask(ForwardTrainer, epochs, architecture, train_dataset, validation_dataset,
                     Loss(PytorchLoss("MSELoss")),
                     Optimizer(PytorchOptimizer("SGD", architecture.parameters(), 0.1)),
                     early_stopping=EarlyStopping(LossMonitor(5, 0.005))).execute()

TestTask(architecture, test_dataset).test(model)
