from PIL import Image

from flogo.data.column import Column


class LoadedImageColumn(Column):

    def __init__(self, values=list(), load: bool = True):
        self.values = [] if len(values) == 0 else values
        super(LoadedImageColumn, self).__init__(self.__load(values) if load else values)

    def __load(self, values):
        return [Image.open(path) for path in values]
