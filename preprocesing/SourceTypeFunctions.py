from pandas import read_csv

from torch.utils.data import DataLoader

from datasets.ImageDataset import ImageDataset
from datasets.NumberDataset import NumberDataset
from preprocesing.ImageProcessor import *


def numbers_source_type_csv(path):
    dataset = read_csv(path, header=None)
    data = dataset.values
    x = data[:, :-1].astype(str)
    y = data[:, -1].astype(str)
    NumberDataset(x, y, "KMf")


def images_source_type(size, mean, std, path):
    transformer = preprocess_images(size, mean, std)
    dataset = ImageDataset(transformer, path)
    dataset_loader = DataLoader(dataset, batch_size=4)
    return dataset_loader


def text_source_type():
    pass
