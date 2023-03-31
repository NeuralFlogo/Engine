import torch

from pytorch.model.sections.link.classification import Classification
from pytorch.model.sections.link.flatten import Flatten
from pytorch.model.sections.processing.convolutional import Convolutional
from pytorch.model.sections.processing.feed_forward import FeedForward
from pytorch.model.models.simple_model import SimpleModel
from pytorch.preprocesing.NumericProcessor import one_hot_encode
from pytorch.preprocesing.SourceTypeFunctions import images_source_type
from pytorch.vocabulary import *

#
# pytorch_architecture = ConvolutionalArchitecture(architecture_conv).pytorch()
# SimpleModel(pytorch_architecture)
#
# architecture_feedforward = [{Channel.In: 10, Channel.Out: 10, Activation.name: "ReLU"},
#                             {Channel.In: 10, Channel.Out: 10, Activation.name: "ReLU"}]
# pytorch_architecture = FeedForward(architecture_feedforward).pytorch()
# SimpleModel(pytorch_architecture)
#
# architecture_classification = {Activation.name: "Softmax", Activation.dimension: 10}
# model.sections.classification.Classification(architecture_classification).pytorch()

# architecture_residual = [{Kernel.Convolutional: (4, 5), Channel.In: 10, Channel.Out: 40, Activation.name: "ReLU",
#                          Kernel.Pool: (2, 2), Stride.Pool: (1, 1), Padding.Pool: (1, 1), Pooling.type: "Max",
#                          Stride.Convolutional: (2, 2), Padding.Convolutional: (1, 1)},
#                         {Kernel.Convolutional: (5, 5), Channel.In: 40, Channel.Out: 40, Activation.name: "ReLU",
#                          Block.HiddenSize: 6, Stride.Convolutional: (1, 1), Padding.Convolutional: (1, 1)},
#                          {Stride.Pool: (2, 2), Padding.Pool: (1, 1), Pooling.type: "Avg",
#                          Stride.Convolutional: (1, 1), Kernel.Pool: (2, 2)}]
# pytorch_architecture = ResNet(architecture_residual).pytorch()

# ResidualModule(pytorch_architecture).forward(torch.zeros((10, 100, 100)))

# architecture_recurrent = {Layers.Size: 3, Block.HiddenSize: 10, Channel.In: 4, Block.Type: "LSTMCell",
#                           Activation.name: "ReLU", Layers.Bias: True, Channel.Out: 2}
# pytorch_architecture = recurrent(architecture_recurrent).pytorch()
# model = RnnModel(pytorch_architecture)
# asdf = model.forward(torch.zeros(4))
# print(asdf[0])

PATH = "/Users/jose_juan/Desktop/training_set_test"
data_loader = images_source_type(224, 0, 1, path=PATH, batch_size=64)

architecture_conv = [{Kernel.Convolutional: 3, Channel.In: 3, Channel.Out: 64, Activation.name: "ReLU",
                      Stride.Convolutional: 1, Padding.Convolutional: 1, Pooling.type: "Max",
                      Kernel.Pool: 2, Stride.Pool: 2, Padding.Pool: 0},
                     {Kernel.Convolutional: 3, Channel.In: 64, Channel.Out: 128, Activation.name: "ReLU",
                      Stride.Convolutional: 1, Padding.Convolutional: 1, Pooling.type: "Max",
                      Kernel.Pool: 2, Stride.Pool: 2, Padding.Pool: 0},
                     {Kernel.Convolutional: 3, Channel.In: 128, Channel.Out: 256, Activation.name: "ReLU",
                      Stride.Convolutional: 1, Padding.Convolutional: 1, Pooling.type: "Max",
                      Kernel.Pool: 2, Stride.Pool: 2, Padding.Pool: 0},
                     {Kernel.Convolutional: 5, Channel.In: 256, Channel.Out: 512, Activation.name: "ReLU",
                      Stride.Convolutional: 1, Padding.Convolutional: 1, Pooling.type: "Max",
                      Kernel.Pool: 2, Stride.Pool: 2, Padding.Pool: 0},
                     {Kernel.Convolutional: 3, Channel.In: 512, Channel.Out: 512, Activation.name: "ReLU",
                      Stride.Convolutional: 1, Padding.Convolutional: (2, 2), Pooling.type: "Max",
                      Kernel.Pool: 2, Stride.Pool: 2, Padding.Pool: 0},
                     {Kernel.Convolutional: 3, Channel.In: 512, Channel.Out: 512, Activation.name: "ReLU",
                      Stride.Convolutional: 1, Padding.Convolutional: 1, Pooling.type: "Max",
                      Kernel.Pool: 2, Stride.Pool: 2, Padding.Pool: 0},
                     {Kernel.Convolutional: 3, Channel.In: 512, Channel.Out: 4096, Activation.name: "ReLU",
                      Stride.Convolutional: 1, Padding.Convolutional: 1, Pooling.type: "Max",
                      Kernel.Pool: 2, Stride.Pool: 2, Padding.Pool: 0}]

architecture_feedforward = [{Channel.In: 4096, Channel.Out: 2000, Activation.name: "ReLU"},
                            {Channel.In: 2000, Channel.Out: 2, Activation.name: "ReLU"}]

architecture_classification = {Activation.name: "Softmax", Activation.dimension: 1}

architecture_link = {Dimension.Start: 1, Dimension.End: 3}

pytorch_architecture = Convolutional(architecture_conv).build()
pytorch_architecture += [Flatten(architecture_link).build()]
pytorch_architecture += FeedForward(architecture_feedforward).build()
pytorch_architecture += [
    Classification(architecture_classification).build()]
model = SimpleModel(pytorch_architecture)
print(model)

# Training(model, data_loader, data_loader, torch.nn.CrossEntropyLoss(), torch.optim.SGD(params=model.parameters(),
#                                                                                       lr=0.001, momentum=0.9), 10)\
#    .train()

num_epochs = 10
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):  # I decided to train the model for 50 epochs
    loss_var = 0

    for idx, (images, labels) in enumerate(data_loader):
        labels = torch.tensor(one_hot_encode(labels))
        optimizer.zero_grad()
        scores = model(images)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        loss_var += loss.item()
        if idx % 64 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}] || Step [{idx + 1}/{len(data_loader)}] || Loss:{loss_var / len(data_loader)}')
    print(f"Loss at epoch {epoch + 1} || {loss_var / len(data_loader)}")
