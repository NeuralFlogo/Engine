from PIL import ImageFilter

from framework.data.dataframe.columns.loaded_image import LoadedImageColumn
from framework.preprocessing.mapper import Mapper


class ImageGaussMapper(Mapper):

    def __init__(self, radius=2):
        self.radius = radius

    def apply(self, column: LoadedImageColumn):
        return LoadedImageColumn([self.__add_gaussian_noise(image) for image in column.values], False)

    def __add_gaussian_noise(self, image):
        return image.filter(ImageFilter.GaussianBlur(radius=self.radius))
