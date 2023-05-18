from torchvision import transforms

from framework.data.dataframe.columns.loaded_image import LoadedImageColumn
from framework.data.dataframe.columns.number import NumericColumn
from framework.preprocessing.mapper import Mapper


class TensorMapper(Mapper):

    def __init__(self):
        self.transform = transforms.ToTensor()

    def apply(self, column: NumericColumn):
        return LoadedImageColumn([self.transform(image) for image in column.values], False)
