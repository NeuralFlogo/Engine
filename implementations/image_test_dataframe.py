import torch
from matplotlib import pyplot as plt

from flogo.preprocessing.columns.loaded_image import LoadedImageColumn
from flogo.preprocessing.map_orchestrator import MapOrchestrator
from flogo.preprocessing.mappers.column_mapper import ColumnMapper
from flogo.preprocessing.mappers.grayscale_mapper import GrayScaleMapper
from flogo.preprocessing.mappers.resize_mapper import ResizeMapper
from flogo.preprocessing.readers.image_reader import ImageReader
from pytorch.preprocessing.mappers.tensor_mapper import TensorMapper

path = "C:/Users/Joel/Desktop/prueba"
dataframe = ImageReader().read(path)
dataframe1 = MapOrchestrator(ColumnMapper(LoadedImageColumn), TensorMapper()).process(dataframe, ["input"], ["input'"])
plt.imshow(torch.tensor(dataframe1.get("input''").values[0]).permute(1, 2, 0))
plt.show()
dataframe = MapOrchestrator(ColumnMapper(LoadedImageColumn), ResizeMapper((50, 50)), GrayScaleMapper(), TensorMapper())\
    .process(dataframe, ["input"], ["input'"], ["input''"], ["input'''"])
plt.imshow(dataframe.get("input''''").values[0].permute(1, 2, 0))
plt.show()