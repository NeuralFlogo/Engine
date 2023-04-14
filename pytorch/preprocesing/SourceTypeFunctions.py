import torch
from pandas import read_csv

from torch.utils.data import DataLoader

from pytorch.datasets.ImagesDataset import ImagesDataset
from pytorch.datasets.NumericDataset import NumericDataset
from pytorch.datasets.WordsDataset import WordsDataset
from pytorch.preprocesing.ImageProcessor import *


def __get_loader(dataset, batch_size=4, proportion=0.8):
    train_size = __get_train_size(dataset, proportion)
    train, test = torch.utils.data.random_split(dataset, [train_size, __get_test_size(dataset, train_size)])
    return DataLoader(train, batch_size=batch_size), DataLoader(test, batch_size=batch_size)


def __get_test_size(dataset, train_size):
    return len(dataset) - train_size


def __get_train_size(dataset, proportion):
    return int(proportion * len(dataset))


def numbers_source_type_csv(path, parameters, batch_size):
    dataset = read_csv(path, header=None)
    data = dataset.values
    x = data[:, 1:-1].astype(str)
    y = data[:, 0].astype(str)
    dataset = NumericDataset(x, y, parameters)
    return __get_loader(dataset, batch_size)


def images_source_type(size, mean, std, path, batch_size):
    transformer = preprocess_images(size, mean, std)
    dataset = ImagesDataset(transformer, path)
    return __get_loader(dataset, batch_size)


def text_source_type(paths):
    dataset = WordsDataset(paths)
    return __get_loader(dataset)
