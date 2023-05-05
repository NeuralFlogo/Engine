import numpy as np
from PIL import Image

from flogo.preprocessing.column import Column


class LoadedImageColumn(Column):

    def __init__(self, values=[], load: bool=True):
        super(LoadedImageColumn, self).__init__(self.__load(values) if load else values)

    def __load(self, values):
        return [Image.open(path) for path in values]

