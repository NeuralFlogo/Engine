from Convolutional import ConvolutionalArchitecture
import Module

architecture = [{"kernel_conv": (5, 5), "in_channels": 10, "out_channels": 10, "activation": "ReLU",
                 "kernel_pool": (2, 2), "stride": (2, 2), "padding": (1, 1)},
                {"kernel_conv": (5, 5), "in_channels": 10, "out_channels": 10, "activation": "ReLU",
                 "kernel_pool": (2, 2), "stride": (2, 2), "padding": (1, 1)},
                {"kernel_conv": (5, 5), "in_channels": 10, "out_channels": 10, "activation": "ReLU",
                 "kernel_pool": (2, 2), "stride": (2, 2), "padding": (1, 1)}]

pytorch_architecture = ConvolutionalArchitecture(architecture).pytorch()
print(Module.Module(pytorch_architecture))
