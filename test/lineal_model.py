from flogo.structure.blocks.classification import ClassificationBlock
from flogo.structure.blocks.linear import LinearBlock
from flogo.structure.layers.activation import Activation
from flogo.structure.layers.classification import Classification
from flogo.structure.layers.linear import Linear
from flogo.training.loss import FlogoLossFunction
from flogo.training.optimizer import FlogoOptimizer
from flogo.training.training import FlogoTraining
from pytorch.architecture.forward import ForwardArchitecture
from pytorch.structure.sections.link.classification import ClassificationSection
from pytorch.structure.sections.processing.feed_forward import FeedForwardSection
from pytorch.preprocesing.SourceTypeFunctions import numbers_source_type_csv
from pytorch.training.test import Testing
from pytorch.training.forward_train import ForwardTraining

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

ForwardTraining(FlogoTraining(EPOCHS, model, training_loader=train_data_loader, validation_loader=train_data_loader,
                              loss_function=FlogoLossFunction("MSELoss"),
                              optimizer=FlogoOptimizer("SGD", model_params=model.parameters(), lr=0.1))).train()

Testing(model, test_data_loader).test()
