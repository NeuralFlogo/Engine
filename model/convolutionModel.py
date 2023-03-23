import torch


class ConvolutionalModel:
    def __init__(self, architecture):
        self.arch = [ConvolutionalBlock(architecture[key]) for key in architecture.keys]


class ConvolutionalBlock:
    def __init__(self, architecture):
        self.kernel_con = architecture["kernel_conv"]
        self.in_channels = architecture["in_channels"]
        self.out_channels = architecture["out_channels"]
        self.activation_function = architecture["activation"]
        self.kernel_pool = architecture["kernel_pool"]



