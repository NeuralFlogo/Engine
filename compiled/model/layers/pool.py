class Pool:
    def __init__(self, kernel, stride, padding, pool_type: str):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.pool_type = pool_type
