from pandas import read_csv

from torch.utils.data import DataLoader

from datasets.ImagesDataset import ImagesDataset
from datasets.NumericDataset import NumericDataset
from datasets.WordsDataset import WordsDataset
from preprocesing.ImageProcessor import *


def get_loader(dataset, batch_size=4):
    return DataLoader(dataset, batch_size=batch_size)


def numbers_source_type_csv(path, parameters):
    dataset = read_csv(path, header=None)
    data = dataset.values
    x = data[:, :-1].astype(str)
    y = data[:, -1].astype(str)
    dataset = NumericDataset(x, y, parameters)
    return get_loader(dataset)


def images_source_type(size, mean, std, path):
    transformer = preprocess_images(size, mean, std)
    dataset = ImagesDataset(transformer, path)
    return get_loader(dataset)


def text_source_type(paths):
    dataset = WordsDataset(paths)
    return get_loader(dataset)
