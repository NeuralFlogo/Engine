class Pool:
    def __init__(self, kernel, stride, padding, pooling_type: str):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.pooling_type = pooling_type
