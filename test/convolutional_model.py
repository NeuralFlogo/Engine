import torch
from torchvision import datasets
from torchvision.transforms import transforms

from flogo.structure.blocks.convolutional import ConvolutionalBlock
from flogo.structure.blocks.flatten import FlattenBlock
from flogo.structure.blocks.linear import LinearBlock
from flogo.structure.layers.activation import Activation
from flogo.structure.layers.convolutional import Convolutional
from flogo.structure.layers.flatten import Flatten
from flogo.structure.layers.linear import Linear
from flogo.structure.layers.pool import Pool
from flogo.training.loss import FlogoLossFunction
from flogo.training.optimizer import FlogoOptimizer
from flogo.training.training import FlogoTraining
from pytorch.architecture.forward import ForwardArchitecture
from pytorch.structure.sections.link.flatten import FlattenSection
from pytorch.structure.sections.processing.convolutional import ConvolutionalSection
from pytorch.structure.sections.processing.feed_forward import FeedForwardSection
from pytorch.training.forward_train import ForwardTraining
from pytorch.training.test import Testing

BATCH_SIZE = 50
PATH = "C:/Users/Joel/Documents/Universidad/TercerCurso/SegundoSemestre/PracticasExternas/Proyectos/food"
EPOCHS = 1

convolutional = [ConvolutionalBlock([
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
    Pool("Max")
])]

flatten = FlattenBlock(Flatten(1, 3))

feed_forward = [LinearBlock([
    Linear(400, 200),
    Activation("ReLU"),
    Linear(200, 101)]
)]

convolutional_section = ConvolutionalSection(convolutional).build()
flatten_section = FlattenSection(flatten).build()
feed_forward_section = FeedForwardSection(feed_forward).build()

model = ForwardArchitecture(convolutional_section + flatten_section + feed_forward_section)

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
