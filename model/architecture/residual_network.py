from model.architecture.layers import Conv2d, Pool, ActivationFunction


class ResNet:
    def __init__(self, architecture):
        body = [BodyBlock(block) for block in architecture[1:-1]]
        self.architecture = [InputBlock(architecture[0])] + body + [OutputBlock(architecture[-1])]

    def pytorch(self):
        return [block.pytorch() for block in self.architecture]


class InputBlock:
    def __init__(self, block):
        self.conv = Conv2d(block["kernel_conv"], block["in_channels"], block["out_channels"],
                           block["stride_conv"], block["padding_conv"])
        self.pool = Pool(block["kernel_pool"], block["stride_pool"],
                         block["padding_pool"], block["pooling_type"])

    def pytorch(self):
        return self.conv.pytorch(), self.pool.pytorch()


class BodyBlock:
    def __init__(self, block):
        self.stages = [ResidualBlock(block) for _ in range(block["block_number"])]

    def pytorch(self):
        return [res_block.pytorch() for res_block in self.stages]


class ResidualBlock:
    def __init__(self, block):
        self.conv1 = Conv2d(block["kernel_conv"], block["in_channels"],
                            block["out_channels"], block["stride"], block["padding"])
        self.activation = ActivationFunction(block["activation"])
        self.conv2 = Conv2d(block["kernel_conv"], block["in_channels"],
                            block["out_channels"], block["stride"], block["padding"])

    def pytorch(self):
        return self.conv1.pytorch(), self.activation.pytorch(), self.conv2.pytorch()


class OutputBlock:
    def __init__(self, block):
        self.activation = Pool(block["kernel_pool"], block["stride_pool"],
                               block["padding_pool"], block["pooling_type"])

    def pytorch(self):
        return self.activation.pytorch()
