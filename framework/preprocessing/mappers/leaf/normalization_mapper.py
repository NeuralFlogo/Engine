import numpy as np

from framework.data.dataframe.columns.number import NumericColumn
from framework.preprocessing.mapper import Mapper


class NormalizationMapper(Mapper):
    def __init__(self, min=0, max=1):
        self.min = min
        self.max = max

    def apply(self, column: NumericColumn):
        column_array = np.array(column.values).astype(float)
        return self.__create_column(((column_array - min(column_array)) / (max(column_array) - min(column_array)) *
                                     (self.max - self.min) + self.min).tolist())

    @staticmethod
    def __create_column(standardized_list):
        return NumericColumn(standardized_list, dtype=float)
