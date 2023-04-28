import os

import numpy as np
from PIL import Image

from pytorch.preprocesing.NumericProcessor import one_hot_encode


class ImageDirectoryProcessor:

    def __init__(self, data_dir, transformer, boolean_shuffle):
        self.data_dir = data_dir
        self.transformer = transformer
        self.labels = []
        self.inputs = []
        self.__extract_data_from_dir()
        if boolean_shuffle:
            self.__shuffle()

    def __shuffle(self):
        indexs = np.arange(len(self.labels))
        np.random.shuffle(indexs)

        self.labels = self.labels[indexs]
        self.inputs = self.inputs[indexs]

    def __extract_data_from_dir(self):
        for directory in os.listdir(self.data_dir):
            for file in os.listdir(os.path.join(self.data_dir, directory)):
                image = self.__read_image(os.path.join(self.data_dir, directory, file))
                self.labels.append(directory)
                self.inputs.append(image)

    def __read_image(self, path):
        image = np.array(Image.open(path))
        return np.array(self.transformer(image))

    def transform_inputs(self):
        return self.inputs

    def transform_outputs(self):
        return one_hot_encode(self.labels).tolist()
