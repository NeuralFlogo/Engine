from model.architecture.layers import Linear, ActivationFunction


class FeedForward:
    def __init__(self, architecture):
        self.architecture = [LinearBlock(block) for block in architecture]

    def pytorch(self):
        result = []
        for block in self.architecture: result.extend(block.pytorch())
        return result


class LinearBlock:
    def __init__(self, block):
        self.linear = Linear(block["input_dimension"], block["output_dimension"])
        self.activation = ActivationFunction(block["activation"])

    def pytorch(self):
        return self.linear.pytorch(), self.activation.pytorch()
