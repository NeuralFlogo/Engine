import torch
from torchvision import datasets
from torchvision.transforms import transforms

from flogo.structure.blocks.convolutional import ConvolutionalBlock
from flogo.structure.blocks.flatten import FlattenBlock
from flogo.structure.blocks.linear import LinearBlock
from flogo.structure.layers.convolutional import Convolutional
from pytorch.architecture.forward import ForwardArchitecture
from flogo.layers.activation import Activation
from flogo.layers.flatten import Flatten
from flogo.layers.linear import Linear
from flogo.layers.pool import Pool
from flogo.discovery.hyperparameters.loss import Loss
from flogo.discovery.hyperparameters.optimizer import Optimizer
from flogo.discovery.training_task import TrainingTask
from pytorch.model.sections.link.flatten import FlattenSection
from pytorch.model.sections.processing.convolutional import ConvolutionalSection
from pytorch.model.sections.processing.feed_forward import FeedForwardSection
from pytorch.discovery.trainers.forward_trainer import ForwardTrainer
from pytorch.discovery.test_task import Testing

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

convolutional_section = ConvolutionalSection(convolutional).__build()
flatten_section = FlattenSection(flatten).__build()
feed_forward_section = FeedForwardSection(feed_forward).__build()

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

# download and transform implementations dataset
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

ForwardTrainer(TrainingTask(EPOCHS, model, training_dataset=train_loader, validation_dataset=train_loader,
                            loss_function=Loss("MSELoss"),
                            optimizer=Optimizer("SGD", model_params=model.parameters(), lr=0.1))).train()
print("Testear")



Testing(model, test_loader).execute()
