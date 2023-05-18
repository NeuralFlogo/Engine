import copy
import unittest

from framework.data.dataframe.columns.categorical import CategoricalColumn
from framework.data.dataframe.columns.number import NumericColumn
from framework.data.dataframe.readers.delimeted_file_reader import DelimitedFileReader
from framework.data.dataframe.readers.image_reader import ImageReader
from testing.utils import abs_path

columns = {"work_year": CategoricalColumn(), "experience_level": CategoricalColumn(),
           "employment_type": CategoricalColumn(),
           "job_title": CategoricalColumn(), "salary": NumericColumn()}


class ReaderTest(unittest.TestCase):

    def test_read_csv_dataframe_with_header(self):
        reader = DelimitedFileReader(",")
        dataframe = reader.read(abs_path("/resources/dataset_with_header.csv"), copy.deepcopy(columns), True)
        self.assertEqual([10, 10, 10, 10, 10], [len(dataframe.get(name)) for name in dataframe.column_names()])

    def test_read_csv_dataframe_without_header(self):
        reader = DelimitedFileReader(",")
        dataframe = reader.read(abs_path("/resources/dataset_without_header.csv"), copy.deepcopy(columns), False)
        self.assertEqual([10, 10, 10, 10, 10], [len(dataframe.get(name)) for name in dataframe.column_names()])

    def test_read_image_dataframe(self):
        reader = ImageReader()
        dataframe = reader.read(abs_path("/resources/image_data/"))
        self.assertEqual([20, 20], [len(dataframe.get(name)) for name in dataframe.column_names()])
