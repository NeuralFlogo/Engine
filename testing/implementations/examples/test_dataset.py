from flogo.data.columns.loaded_image import LoadedImageColumn
from flogo.data.dataset_builder import DatasetBuilder
from flogo.data.readers.image_reader import ImageReader
from flogo.preprocessing.mappers.column_mapper import ColumnMapper
from flogo.preprocessing.mappers.grayscale_mapper import GrayScaleMapper
from flogo.preprocessing.mappers.one_hot_mapper import OneHotMapper
from flogo.preprocessing.mappers.resize_mapper import ResizeMapper
from flogo.preprocessing.orchestrator import Orchestrator
from pytorch.preprocessing.mappers.tensor_mapper import TensorMapper
from pytorch.preprocessing.pytorch_caster import PytorchCaster

path = "C:/Users/Joel/Desktop/prueba"
dataframe = ImageReader().read(path)
dataframe = Orchestrator(OneHotMapper(), ColumnMapper(LoadedImageColumn), ResizeMapper((50, 50)), GrayScaleMapper())\
    .process(dataframe, ["output"], ["input"], ["input'"], ["input''"])

dataset = DatasetBuilder(PytorchCaster()).build(dataframe, ["input'''"], ["output0", "output1"], 3)
