from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer
from flogo.discovery.training_task import TrainingTask
from flogo.structure.blocks.classification import ClassificationBlock
from flogo.structure.blocks.linear import LinearBlock
from flogo.structure.layers.activation import Activation
from flogo.structure.layers.classification import Classification
from flogo.structure.layers.linear import Linear
from pytorch.architecture.forward import ForwardArchitecture
from pytorch.discovery.trainers.forward_trainer import ForwardTrainer
from pytorch.discovery.test_task import Testing
from pytorch.preprocesing.SourceTypeFunctions import numbers_source_type_csv
from pytorch.structure.sections.link.classification import ClassificationSection
from pytorch.structure.sections.processing.feed_forward import FeedForwardSection

BATCH_SIZE = 50
EPOCHS = 20
parameters = ["one-hot"] * 22

feedforward = [LinearBlock([
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
])]

classification = ClassificationBlock(Classification("Softmax", 1))

linear_section = FeedForwardSection(feedforward).build()
classification_section = ClassificationSection(classification).build()

model = ForwardArchitecture(linear_section + classification_section)

train_data_loader, test_data_loader = numbers_source_type_csv("C:/Users/Joel/Desktop/breast cancer/mushrooms.csv",
                                                              parameters, BATCH_SIZE)

ForwardTrainer(TrainingTask(EPOCHS, model, training_loader=train_data_loader, validation_dataset=train_data_loader,
                            loss_function=Loss("MSELoss"),
                            optimizer=Optimizer("SGD", model_params=model.parameters(), lr=0.1))).train()

Testing(model, test_data_loader).execute()
