import torch


class Conv2d:
    def __init__(self, kernel_size, channel_in, channel_out, stride, padding):
        self.kernel_size = kernel_size
        self.in_channels = channel_in
        self.out_channels = channel_out
        self.stride = stride
        self.padding = padding

    def build(self):
        return torch.nn.Conv2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                               in_channels=self.in_channels, out_channels=self.out_channels)
