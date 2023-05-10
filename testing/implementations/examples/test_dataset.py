from flogo.data.columns.loaded_image import LoadedImageColumn
from flogo.data.dataset_builder import DatasetBuilder
from flogo.data.dataset_splitter import DatasetSplitter
from flogo.data.readers.image_reader import ImageReader
from flogo.preprocessing.mappers.composite import CompositeMapper
from flogo.preprocessing.mappers.leaf.grayscale_mapper import GrayScaleMapper
from flogo.preprocessing.mappers.leaf.one_hot_mapper import OneHotMapper
from flogo.preprocessing.mappers.leaf.resize_mapper import ResizeMapper
from flogo.preprocessing.mappers.leaf.type_mapper import TypeMapper

from flogo.preprocessing.orchestrator import Orchestrator
from pytorch.preprocessing.pytorch_caster import PytorchCaster

path = "C:/Users/Joel/Desktop/mnist"
dataframe = ImageReader().read(path)
dataframe = Orchestrator(OneHotMapper(), CompositeMapper([TypeMapper(LoadedImageColumn), ResizeMapper((28, 28))]))\
    .process(dataframe, ["output"], ["input"])

dataset = DatasetBuilder(PytorchCaster()).build(dataframe, ["input'"], ["output_0", "output_1", "output_2", "output_3",
                                                                        "output_4", "output_5", "output_6", "output_7",
                                                                        "output_8", "output_9"], 5)
train_dataset, test_dataset, validation_dataset = DatasetSplitter().split(dataset)
