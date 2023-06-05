from framework.data.dataframe.column import Column
from framework.data.dataframe.columns.loaded_image import LoadedImageColumn
from framework.preprocessing.mapper import Mapper


class ResizeMapper(Mapper):
    def __init__(self, size):
        self.size = size

    def apply(self, column: Column):
        return LoadedImageColumn([image.resize(self.size) for image in column.values], False)
