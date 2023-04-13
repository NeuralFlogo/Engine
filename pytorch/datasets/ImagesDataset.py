import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

from pytorch.preprocesing.NumericProcessor import one_hot_encode


def get_label(file_name):
    return file_name.split(".")[0]


def transform_files(files):
    labels = []
    for file in files:
        labels.append(get_label(file))
    return np.array(labels)


class ImagesDataset(Dataset):
    def __init__(self, transformer, data_dir):
        self.transformer = transformer
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir)]
        self.labels = one_hot_encode(transform_files(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        label = self.labels[idx]
        image = read_image(os.path.join(self.data_dir, file_name), mode=ImageReadMode.RGB)
        image = torch.tensor(image, dtype=torch.float32)
        image = self.transformer(image)
        return image, torch.tensor(label, dtype=torch.float32)
