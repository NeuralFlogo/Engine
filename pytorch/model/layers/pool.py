from torch.nn import MaxPool2d, AvgPool2d


class Pool:
    def __init__(self, kernel, stride, padding, pool_type):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.pooling_type = pool_type

    def build(self):
        if self.pooling_type == "Max":
            return MaxPool2d(kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        if self.pooling_type == "Avg":
            return AvgPool2d(kernel_size=self.kernel, stride=self.stride, padding=self.padding)
