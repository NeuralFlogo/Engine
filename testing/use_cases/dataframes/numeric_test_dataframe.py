import os

from framework.data.dataframe.columns.categorical import CategoricalColumn
from framework.data.dataframe.columns.number import NumericColumn
from framework.data.dataframe.readers.delimeted_file_reader import DelimitedFileReader
from framework.preprocessing.mappers.leaf.normalization_mapper import NormalizationMapper
from framework.preprocessing.mappers.leaf.one_hot_mapper import OneHotMapper
from framework.preprocessing.mappers.leaf.standarization_mapper import StandardizationMapper
from framework.preprocessing.orchestrator import Orchestrator


def abs_path(part_path):
    return os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))) + part_path


path = abs_path("/resources/dataset_with_header.csv")

columns = {
    "work_year": CategoricalColumn(),
    "experience_level": CategoricalColumn(),
    "employment_type": CategoricalColumn(),
    "job_title": CategoricalColumn(),
    "salary": NumericColumn()}

dataframe = DelimitedFileReader(",").read(path, columns)

dataframe = Orchestrator(OneHotMapper(), NormalizationMapper(min=-1, max=1), StandardizationMapper()) \
    .process(dataframe,
             ["work_year", "experience_level", "employment_type", "job_title"],
             ["salary"],
             ["salary'"])

print(dataframe.column_names())
