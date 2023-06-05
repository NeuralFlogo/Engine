class Pool:
    def __init__(self, pool_type: str, kernel=2, stride=2, padding=0):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.pool_type = pool_type
