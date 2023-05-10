from torchvision import transforms

from flogo.data.columns.loaded_image import LoadedImageColumn
from flogo.data.columns.number import NumericColumn
from flogo.preprocessing.mapper import Mapper


class TensorMapper(Mapper):

    def __init__(self):
        self.transform = transforms.ToTensor()

    def apply(self, column: NumericColumn):
        return LoadedImageColumn([self.transform(image) for image in column.values], False)
