import numpy as np

from framework.data.dataframe.columns.number import NumericColumn
from framework.preprocessing.mapper import Mapper


class NumberGaussianNoiseMapper(Mapper):

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def apply(self, column: NumericColumn):
        column_array = np.array(column.values).astype(float)
        return self.__create_column((column_array + self.__gaussian_noise(column_array)).tolist())

    def __gaussian_noise(self, column_array):
        return np.random.normal(self.mean, self.std, column_array.shape[0])

    @staticmethod
    def __create_column(gaussian_list: list):
        return NumericColumn(gaussian_list, dtype=float)