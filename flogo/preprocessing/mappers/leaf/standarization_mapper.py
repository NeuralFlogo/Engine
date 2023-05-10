import numpy as np

from flogo.data.columns.number import NumericColumn
from flogo.preprocessing.mapper import Mapper


class StandardizationMapper(Mapper):
    def apply(self, column: NumericColumn):
        column_array = np.array(column.values).astype(float)
        return self.__create_column((column_array - np.mean(column_array)) / np.std(column_array).tolist())

    @staticmethod
    def __create_column(standardized_list):
        return NumericColumn(standardized_list, dtype=float)
