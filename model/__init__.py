from Convolutional import ConvolutionalArchitecture
import Module
from model.FeedForward import FeedForward

architecture_conv = [{"kernel_conv": (5, 5), "in_channels": 10, "out_channels": 10, "activation": "ReLU",
                      "kernel_pool": (2, 2), "stride": (2, 2), "padding": (1, 1)},
                     {"kernel_conv": (5, 5), "in_channels": 10, "out_channels": 10, "activation": "ReLU",
                      "kernel_pool": (2, 2), "stride": (2, 2), "padding": (1, 1)},
                     {"kernel_conv": (5, 5), "in_channels": 10, "out_channels": 10, "activation": "ReLU",
                      "kernel_pool": (2, 2), "stride": (2, 2), "padding": (1, 1)}]

pytorch_architecture = ConvolutionalArchitecture(architecture_conv).pytorch()
print(Module.Module(pytorch_architecture))


architecture_feedforward = [{"input_dimension": 10, "output_dimension": 10, "activation": "ReLU"}
                            , {"input_dimension": 10, "output_dimension": 10, "activation": "ReLU"}]


pytorch_architecture = FeedForward(architecture_feedforward).pytorch()
print(Module.Module(pytorch_architecture))