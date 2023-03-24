from model.layers import ActivationFunction, Pool, Conv2d


class ConvolutionalArchitecture:
    def __init__(self, architecture):
        self.architecture = [ConvolutionalBlock(block) for block in architecture]

    def pytorch(self):
        result = []
        for block in self.architecture: result.extend(block.pytorch())
        return result


class ConvolutionalBlock:
    def __init__(self, block):
        self.convolutional_layer = Conv2d(block["kernel_conv"],
                                          block["in_channels"],
                                          block["out_channels"],
                                          block["stride_pool"],
                                          block["padding_pool"])
        self.activation_function = ActivationFunction(block["activation"])
        self.pool_layer = Pool(block["kernel_pool"],
                               block["stride_pool"],
                               block["padding_pool"],
                               block["pooling_type"])

    def pytorch(self):
        return self.convolutional_layer.pytorch(), self.activation_function.pytorch(), self.pool_layer.pytorch()
