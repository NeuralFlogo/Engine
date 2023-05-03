import os

import numpy as np
from PIL import Image

from pytorch.preprocessing.preprocessors.images.image_adapter import ImageAdapter
from pytorch.preprocessing.preprocessors.numbers.one_hot_preprocessor import OneHotPreprocessor


class ImageDirectoryPreprocessor:

    def __init__(self, size, mean=[0, 0, 0], std=[1, 1, 1]):
        self.image_adapter = ImageAdapter(size, mean, std).adaptions()
        self.inputs = []
        self.labels = []

    def process(self, data_path, shuffle):
        for directory in os.listdir(data_path):
            self.extract_images_from(data_path, directory)
        self.labels = OneHotPreprocessor.process(self.labels)
        self.inputs = np.array(self.inputs)
        if shuffle:
            self.__shuffle()
        return self.inputs, self.labels

    def extract_images_from(self, data_path, directory,):
        for file in os.listdir(os.path.join(data_path, directory)):
            image = self.__read_image(os.path.join(data_path, directory, file))
            self.labels.append(directory)
            self.inputs.append(image)

    def __read_image(self, path):
        image = np.array(Image.open(path))
        return self.adapt_image(image)

    def adapt_image(self, image):
        return np.array(self.image_adapter(image))

    def __shuffle(self):
        index = np.random.permutation(len(self.labels))
        self.inputs, self.labels = self.inputs[index], self.labels[index].tolist()
