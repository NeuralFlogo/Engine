from torch.utils.data import DataLoader

from datasets.ImageDataset import ImageDataset
from preprocesing.ImageProcessor import *


def numbers_source_type():
    pass


def images_source_type(size, mean, std, path):
    transformer = preprocess_images(size, mean, std)
    dataset = ImageDataset(path, transformer)
    dataset_loader = DataLoader(dataset, batch_size=4)
    return dataset_loader


def text_source_type():
    pass
