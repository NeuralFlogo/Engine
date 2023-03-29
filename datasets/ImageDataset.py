import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode


def get_label(file_name):
    return file_name.split(".")[0]


class ImageDataset(Dataset):
    def __init__(self, transformer, data_dir):
        self.transformer = transformer
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        label = get_label(file_name)
        image = read_image(os.path.join(self.data_dir, file_name), mode=ImageReadMode.RGB)
        image = torch.tensor(image, dtype=torch.float32)
        image = self.transformer(image)
        return image, label