import os
import unittest

from numpy import std

from framework.data.dataframe.columns.categorical import CategoricalColumn
from framework.data.dataframe.columns.loaded_image import LoadedImageColumn
from framework.data.dataframe.columns.number import NumericColumn
from framework.data.dataframe.readers.delimeted_file_reader import DelimitedFileReader
from framework.data.dataframe.readers.image_reader import ImageReader
from framework.preprocessing.mappers.composite import CompositeMapper
from framework.preprocessing.mappers.leaf.number_gauss_mapper import GaussMapper
from framework.preprocessing.mappers.leaf.type_mapper import TypeMapper
from framework.preprocessing.orchestrator import Orchestrator


def abs_path(part_path):
    return os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))) + part_path


class GaussMapperTest(unittest.TestCase):

    def test_gauss_mapper_on_numbers(self):
        columns = {"Gender": CategoricalColumn(), "Ethnicity": CategoricalColumn(), "ParentLevel": CategoricalColumn(),
                   "Lunch": CategoricalColumn(), "Test": CategoricalColumn(),
                   "Math": NumericColumn(dtype=int), "Reading": NumericColumn(dtype=int),
                   "Writing": NumericColumn(dtype=int)}
        dataframe = DelimitedFileReader(",").read(abs_path("/resources/students_performance_dataset.csv"), columns)
        gauss_dataframe = Orchestrator(GaussMapper(0, 10)).process(dataframe, ["Math", "Reading", "Writing"])
        self.assertNotEqual(std(dataframe.get("Math").values), std(gauss_dataframe.get("Math'").values))

    def test_gauss_mapper_on_images(self):
        path = abs_path("/resources/mnist")
        dataframe = ImageReader().read(path)
        gauss_dataframe = Orchestrator(CompositeMapper([TypeMapper(LoadedImageColumn), GaussMapper(0, 10)]))\
            .process(dataframe, ["input"])
