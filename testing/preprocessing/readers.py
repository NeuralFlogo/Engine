import os
import unittest

from framework.data.dataframe.columns.categorical import CategoricalColumn
from framework.data.dataframe.columns.number import NumericColumn
from framework.data.dataframe.readers.delimeted_file_reader import DelimitedFileReader
from framework.data.dataframe.readers.image_reader import ImageReader
from framework.data.timeline.parser import Parser
from framework.data.timeline.readers.timeline_reader import TimelineReader


def abs_path(part_path):
    return os.path.dirname(os.path.abspath(os.getcwd())) + part_path


delimited_file_columns_with_header = {"work_year": CategoricalColumn(), "experience_level": CategoricalColumn(),
                                      "employment_type": CategoricalColumn(), "job_title": CategoricalColumn(),
                                      "salary": NumericColumn()}

delimited_file_columns_without_header = {"work_year": CategoricalColumn(), "experience_level": CategoricalColumn(),
                                         "employment_type": CategoricalColumn(), "job_title": CategoricalColumn(),
                                         "salary": NumericColumn()}


class ReaderTest(unittest.TestCase):

    def test_delimited_file_reader_with_header(self):
        path = abs_path("/resources/dataset_with_header.csv")
        dataframe = DelimitedFileReader(",").read(path, delimited_file_columns_with_header)
        expected = [10, 10, 10, 10, 10]
        self.assertEqual(expected, [len(dataframe.get(column)) for column in dataframe.column_names()])

    def test_delimited_file_reader_without_header(self):
        path = abs_path("/resources/dataset_without_header.csv")
        dataframe = DelimitedFileReader(",").read(path, delimited_file_columns_without_header, header=False)
        expected = [10, 10, 10, 10, 10]
        self.assertEqual(expected, [len(dataframe.get(column)) for column in dataframe.column_names()])

    def test_file_system_image_reader(self):
        path = abs_path("/resources/image_data")
        dataframe = ImageReader().read(path)
        expected = ["input", "output", 20, 20]
        returned = dataframe.column_names() + [len(dataframe.get(column)) for column in dataframe.column_names()]
        self.assertEqual(expected, returned)

    def test_timeline_reader(self):
        path = abs_path("/resources/kraken.its")
        timeline = TimelineReader(Parser()).read(path).to_dataframe()
        expected = [901672, 901672, 901672, 901672, 901672, 901672, 901672, 901672, 901672, 901672, 901672, 901672]
        self.assertEqual(expected, [len(timeline.get(column)) for column in timeline.column_names()])
