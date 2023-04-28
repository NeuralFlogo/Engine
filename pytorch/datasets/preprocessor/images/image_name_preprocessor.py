import os

import numpy as np
from PIL import Image

from pytorch.datasets.preprocessor.images.image_adapter import ImageAdapter
from pytorch.datasets.preprocessor.numbers.one_hot_preprocessor import OneHotPreprocessor


def get_label(file_name):
    return file_name.split(".")[0]


def get_labels(files):
    labels = []
    for file in files:
        labels.append(get_label(file))
    return np.array(labels)


class ImageNamePreprocessor:

    def __init__(self, size, mean=[0, 0, 0], std=[1, 1, 1]):
        self.image_adapter = ImageAdapter(size, mean, std).adaptions()

    def process(self, data_path, boolean_shuffle):
        files = np.array([f for f in os.listdir(data_path)])
        labels = OneHotPreprocessor.process(get_labels(files))
        return self.__shuffle(self.__get_inputs(files, data_path), labels) if boolean_shuffle else \
                   self.__get_inputs(files, data_path), labels.tolist()

    def __get_inputs(self, files, data_path):
        inputs = []
        for file in files:
            image = np.array(Image.open(os.path.join(data_path, file)))
            image = np.array(self.image_adapter(image))
            inputs.append(image)
        return inputs

    @staticmethod
    def __shuffle(inputs, labels):
        index = np.random.permutation(len(labels))
        return inputs[index], labels[index].tolist()
