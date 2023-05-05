from PIL import ImageOps

from flogo.preprocessing.columns.loaded_image import LoadedImageColumn
from flogo.preprocessing.mapper import Mapper


class GrayScaleMapper(Mapper):

    def apply(self, column: LoadedImageColumn):
        return LoadedImageColumn([ImageOps.grayscale(image) for image in column.values], False)
