class Conv:
    def __init__(self, channel_in: int, channel_out: int, kernel=3, stride=1, padding=0):
        self.kernel = kernel
        self.in_channels = channel_in
        self.out_channels = channel_out
        self.stride = stride
        self.padding = padding
