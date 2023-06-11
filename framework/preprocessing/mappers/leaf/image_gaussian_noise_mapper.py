import random

from framework.data.dataframe.columns.loaded_image import LoadedImageColumn
from framework.preprocessing.mapper import Mapper


class ImageGaussianNoiseMapper(Mapper):
    def __init__(self, noise_proportion=None):
        self.noise_proportion = noise_proportion

    def apply(self, column: LoadedImageColumn):
        return LoadedImageColumn([self.__add_gaussian_noise(image.copy()) for image in column.values], False)

    def __add_gaussian_noise(self, image):
        for _ in range(round(self.__total_pixels(image) * self.__noise_proportion())):
            image.putpixel(self.__pixel_to_modify(image), self.__gaussian_noise())
        return image

    def __noise_proportion(self):
        return random.random() if not self.noise_proportion else self.noise_proportion

    def __total_pixels(self, image):
        return self.__image_height(image) * self.__image_width(image)

    def __image_height(self, image):
        return image.size[0]

    def __image_width(self, image):
        return image.size[1]

    def __pixel_to_modify(self, image):
        return random.randint(0, self.__image_height(image) - 1), random.randint(0, self.__image_width(image) - 1)

    def __gaussian_noise(self):
        return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
