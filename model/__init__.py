import torch
from torch import nn

from model.architecture.convolutional import ConvolutionalArchitecture
from model.architecture.residual_network import ResNet
from model.initialize import initialize
from model.models.rnn_model import RnnModel
from model.models.simple_model import SimpleModel
from vocabulary import *
from model.architecture.RNN.rnn import RNN

# architecture_conv = [{Kernel.Convolutional: (5, 5), Channel.In: 10, Channel.Out: 10, Activation.name: "ReLU",
#                       Kernel.Pool: (2, 2), Stride.Pool: (2, 2), Padding.Pool: (1, 1), Pooling.type: "Avg",
#                       Stride.Convolutional: (2, 2), Padding.Convolutional: (1, 1)},
#                      {Kernel.Convolutional: (5, 1), Channel.In: 10, Channel.Out: 10, Activation.name: "ReLU",
#                       Kernel.Pool: (2, 2), Stride.Pool: (2, 2), Padding.Pool: (1, 1), Pooling.type: "Avg",
#                       Stride.Convolutional: (2, 2), Padding.Convolutional: (1, 1)},
#                      {Kernel.Convolutional: (5, 6), Channel.In: 10, Channel.Out: 10, Activation.name: "ReLU",
#                       Kernel.Pool: (2, 2), Stride.Pool: (2, 2), Padding.Pool: (1, 1), Pooling.type: "Avg",
#                       Stride.Convolutional: (2, 2), Padding.Convolutional: (1, 1)}]

# pytorch_architecture = ConvolutionalArchitecture(architecture_conv).pytorch()
# model = SimpleModel(pytorch_architecture)
# initialize(model, nn.init.normal_, mean=0, std=1)
#
# architecture_feedforward = [{Channel.In: 10, Channel.Out: 10, Activation.name: "ReLU"},
#                             {Channel.In: 10, Channel.Out: 10, Activation.name: "ReLU"}]
# pytorch_architecture = FeedForward(architecture_feedforward).pytorch()
# SimpleModel(pytorch_architecture)
#
# architecture_classification = {Activation.name: "Softmax", Activation.dimension: 10}
# model.architecture.classification.Classification(architecture_classification).pytorch()
#
architecture_residual = [{Kernel.Convolutional: (4, 5), Channel.In: 10, Channel.Out: 10, Activation.name: "ReLU",
                          Kernel.Pool: (2, 2), Stride.Pool: (2, 2), Padding.Pool: (1, 1), Pooling.type: "Max",
                          Stride.Convolutional: (2, 2), Padding.Convolutional: (1, 1)},
                         {Kernel.Convolutional: (5, 5), Channel.In: 10, Channel.Out: 10, Activation.name: "ReLU",
                          Block.HiddenSize: 6, Stride.Convolutional: (2, 2), Padding.Convolutional: (1, 1)},
                         {Stride.Pool: (2, 2), Padding.Pool: (1, 1), Pooling.type: "Avg",
                          Stride.Convolutional: (2, 2), Kernel.Pool: (2, 2)}]
print(len(ResNet(architecture_residual).pytorch()[1]))

# architecture_recurrent = {Layers.Size: 3, Block.HiddenSize: 10, Channel.In: 4, Block.Type: "LSTMCell",
#                           Activation.name: "ReLU", Layers.Bias: True, Channel.Out: 2}
# pytorch_architecture = RNN(architecture_recurrent).pytorch()
# model = RnnModel(pytorch_architecture)
# asdf = model.forward(torch.zeros(4))

