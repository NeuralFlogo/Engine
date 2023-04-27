from torch.nn import MaxPool2d, AvgPool2d

import flogo.structure.layers.pool


class PPool:
    def __init__(self, pool: flogo.structure.layers.pool.Pool):
        self.kernel = pool.kernel
        self.stride = pool.stride
        self.padding = pool.padding
        self.pooling_type = pool.pool_type

    def build(self):
        if self.pooling_type == "Max":
            return MaxPool2d(kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        if self.pooling_type == "Avg":
            return AvgPool2d(kernel_size=self.kernel, stride=self.stride, padding=self.padding)
