# from Convolutional import ConvolutionalArchitecture
# import Module
# from main import FeedForward
from main.Engine import engine

architecture_conv = [{"kernel_conv": (5, 5), "in_channels": 10, "out_channels": 10, "activation": "ReLU",
                      "kernel_pool": (2, 2), "stride": (2, 2), "padding": (1, 1)},
                     {"kernel_conv": (5, 5), "in_channels": 10, "out_channels": 10, "activation": "ReLU",
                      "kernel_pool": (2, 2), "stride": (2, 2), "padding": (1, 1)},
                     {"kernel_conv": (5, 5), "in_channels": 10, "out_channels": 10, "activation": "ReLU",
                      "kernel_pool": (2, 2), "stride": (2, 2), "padding": (1, 1)}]

architecture_feedforward = [{"input_dimension": 10, "output_dimension": 10, "activation": "ReLU"}
                            , {"input_dimension": 10, "output_dimension": 10, "activation": "ReLU"}]

architecture = {
    "convolutional": architecture_conv,
    "feedforward": architecture_feedforward
}

# engine(architecture=architecture)
# pytorch_architecture = ConvolutionalArchitecture(architecture_conv).pytorch()
# print(Module.Module(pytorch_architecture))
#
# pytorch_architecture = FeedForward(architecture_feedforward).pytorch()
# print(Module.Module(pytorch_architecture))