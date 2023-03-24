from Convolutional import ConvolutionalArchitecture
from Models import SimpleModel
from model.FeedForward import FeedForward
from RNN.Recurrent import Recurrent

architecture_conv = [{"kernel_conv": (5, 5), "in_channels": 10, "out_channels": 10, "activation": "ReLU",
                      "kernel_pool": (2, 2), "stride": (2, 2), "padding": (1, 1)},
                     {"kernel_conv": (5, 5), "in_channels": 10, "out_channels": 10, "activation": "ReLU",
                      "kernel_pool": (2, 2), "stride": (2, 2), "padding": (1, 1)},
                     {"kernel_conv": (5, 5), "in_channels": 10, "out_channels": 10, "activation": "ReLU",
                      "kernel_pool": (2, 2), "stride": (2, 2), "padding": (1, 1)}]
pytorch_architecture = ConvolutionalArchitecture(architecture_conv).pytorch()
SimpleModel(pytorch_architecture)



architecture_feedforward = [{"input_dimension": 10, "output_dimension": 10, "activation": "ReLU"},
                            {"input_dimension": 10, "output_dimension": 10, "activation": "ReLU"}]
pytorch_architecture = FeedForward(architecture_feedforward).pytorch()
SimpleModel(pytorch_architecture)



architecture_recurrent = {"num_layers": 10, "hidden_size": 4, "input_size": 10, "block_type": "GRUCell",
                          "activation": "ReLU", "bias": True, "output_size": 2}
pytorch_architecture = Recurrent(architecture_recurrent).pytorch()
