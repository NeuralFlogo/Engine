import torch
from torchvision import datasets
from torchvision.transforms import transforms

from model.flogo.blocks.convolutional import FlogoConvolutionalBlock
from model.flogo.blocks.flatten import FlogoFlattenBlock
from model.flogo.blocks.linear import FlogoLinearBlock
from model.flogo.layers.activation import Activation
from model.flogo.layers.convolutional import Conv
from model.flogo.layers.flatten import Flatten
from model.flogo.layers.linear import Linear
from model.flogo.layers.pool import Pool
from model.flogo.training.loss import FlogoLossFunction
from model.flogo.training.optimizer import FlogoOptimizer
from model.flogo.training.training import FlogoTraining
from pytorch.model.models.combination import CombinationModule
from pytorch.model.models.forward import ForwardModule
from pytorch.model.sections.link.flatten import FlattenSection
from pytorch.model.sections.processing.convolutional import ConvolutionalSection
from pytorch.model.sections.processing.feed_forward import FeedForwardSection
from pytorch.training.forward_train import ForwardTraining
from pytorch.training.test import Testing

BATCH_SIZE = 50
PATH = "C:/Users/Joel/Documents/Universidad/TercerCurso/SegundoSemestre/PracticasExternas/Proyectos/food"
EPOCHS = 1

convolutional = [FlogoConvolutionalBlock([
    Conv(3, 50),
    Activation("ReLU"),
    Conv(50, 100),
    Activation("ReLU"),
    Pool("Max"),
    Conv(100, 250),
    Activation("ReLU"),
    Conv(250, 500),
    Activation("ReLU"),
    Pool("Max"),
    Conv(500, 250),
    Activation("ReLU"),
    Conv(250, 100),
    Activation("ReLU"),
    Pool("Max")
])]

flatten = FlogoFlattenBlock(Flatten(1, 3))

feed_forward = [FlogoLinearBlock([
    Linear(400, 200),
    Activation("ReLU"),
    Linear(200, 101)]
)]

convolutional_section = ConvolutionalSection(convolutional).build()
flatten_section = FlattenSection(flatten).build()
feed_forward_section = FeedForwardSection(feed_forward).build()

model = ForwardModule(convolutional_section + flatten_section + feed_forward_section)

train_loader = torch.utils.data.DataLoader(datasets.food101.Food101(PATH,
                                                                    download=True,
                                                                    transform=transforms.Compose([
                                                                        transforms.ToTensor(),
                                                                        transforms.Resize(size=(50, 50)),
                                                                        # first, convert image to PyTorch tensor
                                                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                                             [0.229, 0.224, 0.225])

                                                                        # normalize inputs
                                                                    ])),
                                           batch_size=100,
                                           shuffle=True)

# download and transform test dataset
test_loader = torch.utils.data.DataLoader(datasets.food101.Food101(PATH,
                                                                   download=True,
                                                                   transform=transforms.Compose([
                                                                       transforms.ToTensor(),
                                                                       transforms.Resize(size=(50, 50)),
                                                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                                       # normalize inputs
                                                                   ])),
                                          batch_size=100,
                                          shuffle=True)

ForwardTraining(FlogoTraining(EPOCHS, model, training_loader=train_loader, validation_loader=train_loader,
                              loss_function=FlogoLossFunction("MSELoss"),
                              optimizer=FlogoOptimizer("SGD", model_params=model.parameters(), lr=0.1))).train()
print("Testear")



Testing(model, test_loader).test()
