class Conv:
    def __init__(self, kernel, channel_in: int, channel_out: int, stride, padding):
        self.kernel = kernel
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding
