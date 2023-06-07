import os
import unittest

from framework.data.dataframe.columns.categorical import CategoricalColumn
from framework.data.dataframe.columns.number import NumericColumn
from framework.data.dataframe.readers.delimeted_file_reader import DelimitedFileReader
from framework.data.dataset.dataset_builder import DatasetBuilder
from pytorch.data.torch_gpu_entry_allocator import TorchGpuEntryAllocator
from pytorch.preprocessing.pytorch_caster import PytorchCaster


def abs_path(part_path):
    return os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))) + part_path


delimited_file_columns_with_header = {"work_year": NumericColumn(), "experience_level": CategoricalColumn(),
                                      "employment_type": CategoricalColumn(), "job_title": CategoricalColumn(),
                                      "salary": NumericColumn()}

path = abs_path("/resources/dataset_with_header.csv")
dataframe = DelimitedFileReader(",").read(path, delimited_file_columns_with_header)
dataset = DatasetBuilder(PytorchCaster()).build(dataframe, ["work_year"], ["salary"])


class GpuAllocator(unittest.TestCase):

    def test_allocator(self):
        TorchGpuEntryAllocator().allocate(dataset[0])
        expected = "cuda"
        self.assertEqual(expected, dataset[0].get_input().device.type)

    def test_deallocator(self):
        TorchGpuEntryAllocator().deallocate(dataset[0])
        expected = "cpu"
        self.assertEqual(expected, dataset[0].get_input().device.type)
