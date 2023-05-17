import PIL
import numpy as np
import torch
from torchvision import transforms


class PytorchCaster:

    def __init__(self):
        self.transform = transforms.ToTensor()

    def cast(self, values):
        if type(values[0]) == PIL.Image.Image:
            return torch.stack([self.transform(value) for value in values])
        return torch.Tensor(np.array(values)) if type(values[0]) == list else torch.Tensor(np.array(values)).unsqueeze(1)
