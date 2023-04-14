from model.flogo.blocks.classification import FlogoClassificationBlock
from model.flogo.blocks.linear import FlogoLinearBlock
from model.flogo.layers.activation import Activation
from model.flogo.layers.classification import Classification
from model.flogo.layers.linear import Linear
from model.flogo.training.loss import FlogoLossFunction
from model.flogo.training.flogooptimizer import FlogoOptimizer
from model.flogo.training.flogotraining import FlogoTraining
from pytorch.model.models.simple_model import SimpleModel
from pytorch.model.sections.link.classification import ClassificationSection
from pytorch.model.sections.processing.feed_forward import FeedForwardSection
from pytorch.preprocesing.SourceTypeFunctions import numbers_source_type_csv
from pytorch.training.test import Testing
from pytorch.training.train import Training

BATCH_SIZE = 50
EPOCHS = 200
parameters = ["one-hot"] * 22

feedforward = [FlogoLinearBlock([
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

classification = FlogoClassificationBlock(Classification("Softmax", 1))

linear_section = FeedForwardSection(feedforward).build()
classification_section = ClassificationSection(classification).build()

model = SimpleModel(linear_section + [classification_section])

train_data_loader, test_data_loader = numbers_source_type_csv("C:/Users/Joel/Desktop/breast cancer/mushrooms.csv",
                                                              parameters, BATCH_SIZE)


Training(FlogoTraining(EPOCHS, model, training_loader=train_data_loader, validation_loader=train_data_loader,
                       loss_function=FlogoLossFunction("MSELoss"),
                       optimizer=FlogoOptimizer("SGD", model_params=model.parameters(), lr=0.1))).train()

Testing(model, test_data_loader).test()
