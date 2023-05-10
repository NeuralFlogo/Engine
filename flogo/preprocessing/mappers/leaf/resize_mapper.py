from flogo.data.column import Column
from flogo.data.columns.loaded_image import LoadedImageColumn
from flogo.preprocessing.mapper import Mapper


class ResizeMapper(Mapper):
    def __init__(self, size):
        self.size = size

    def apply(self, column: Column):
        return LoadedImageColumn([image.resize(self.size) for image in column.values], False)
