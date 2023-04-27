import os

import numpy as np
from pytorch.preprocesing.NumericProcessor import one_hot_encode
from PIL import Image


def get_label(file_name):
    return file_name.split(".")[0]


def get_labels(files):
    labels = []
    for file in files:
        labels.append(get_label(file))
    return np.array(labels)


class ImageNameNumpyTransformer:
    def __init__(self, data_dir, transformer, boolean_shuffle):
        self.data_dir = data_dir
        self.files = np.array([f for f in os.listdir(data_dir)])
        if boolean_shuffle:
            np.random.shuffle(self.files)
        self.labels = one_hot_encode(get_labels(self.files))
        self.transformer = transformer

    def transform_inputs(self):
        inputs = []
        for file in self.files:
            image = np.array(Image.open(os.path.join(self.data_dir, file)))
            image = np.array(self.transformer(image))
            inputs.append(image)
        return inputs

    def transform_outputs(self):
        return self.labels.tolist()