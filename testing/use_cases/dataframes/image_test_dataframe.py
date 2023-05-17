import os

import torch
from matplotlib import pyplot as plt

from flogo.data.dataframe.columns.loaded_image import LoadedImageColumn
from flogo.data.dataframe.readers.image_reader import ImageReader
from flogo.preprocessing.mappers.composite import CompositeMapper
from flogo.preprocessing.mappers.leaf.grayscale_mapper import GrayScaleMapper
from flogo.preprocessing.mappers.leaf.resize_mapper import ResizeMapper
from flogo.preprocessing.mappers.leaf.type_mapper import TypeMapper
from flogo.preprocessing.orchestrator import Orchestrator
from pytorch.preprocessing.mappers.tensor_mapper import TensorMapper


def abs_path(part_path):
    return os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))) + part_path


path = abs_path("/resources/image_data")

dataframe = ImageReader().read(path)
dataframe1 = Orchestrator(CompositeMapper([TypeMapper(LoadedImageColumn), TensorMapper()])).process(dataframe, ["input"])
plt.imshow(torch.tensor(dataframe1.get("input'").get_values()[1]).permute(1, 2, 0))
plt.show()

dataframe = Orchestrator(CompositeMapper([TypeMapper(LoadedImageColumn), ResizeMapper((50, 50)), GrayScaleMapper(),
                                          TensorMapper()])).process(dataframe, ["input"])
plt.imshow(dataframe.get("input'").get_values()[1].permute(1, 2, 0))
plt.show()
