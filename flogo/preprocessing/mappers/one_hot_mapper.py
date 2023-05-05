import numpy as np

from flogo.preprocessing.columns.categorical import CategoricalColumn
from flogo.preprocessing.columns.number import NumericColumn
from flogo.preprocessing.mapper import Mapper


class OneHotMapper(Mapper):

    def apply(self, column: CategoricalColumn):
        unique, inverse = np.unique(column.values, return_inverse=True)
        return self.__create_columns(np.eye(unique.shape[0])[inverse])

    @staticmethod
    def __create_columns(one_hot_array):
        return [NumericColumn(one_hot_array[:, index]) for index in range(one_hot_array.shape[1])]
