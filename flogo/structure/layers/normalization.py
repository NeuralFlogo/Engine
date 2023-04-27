class Normalization:
    def __init__(self, out_channels, momentum=0.1, eps=1e-5):
        self.out_channels = out_channels
        self.momentum = momentum
        self.eps = eps
