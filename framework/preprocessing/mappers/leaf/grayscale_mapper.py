from PIL import ImageOps

from framework.data.dataframe.columns.loaded_image import LoadedImageColumn
from framework.preprocessing.mapper import Mapper


class GrayScaleMapper(Mapper):
    def apply(self, column: LoadedImageColumn):
        return LoadedImageColumn([ImageOps.grayscale(image) for image in column.get_values()], False)
