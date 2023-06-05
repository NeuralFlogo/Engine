import numpy as np

from framework.data.dataframe.columns.categorical import CategoricalColumn
from framework.data.dataframe.columns.number import NumericColumn
from framework.data.dataframe.dataframe import Dataframe
from framework.preprocessing.mapper import Mapper


class OneHotMapper(Mapper):
    def map(self, input: Dataframe, indexes):
        output = Dataframe()
        for column_name in indexes:
            categories, columns = self.apply(input.get(column_name))
            output.append_columns(self.__get_new_column_names(column_name, categories), columns)
        output.update(input)
        return output

    def apply(self, column: CategoricalColumn):
        categories, inverse = np.unique(column.values, return_inverse=True)
        return categories, self.__create_columns(np.eye(categories.shape[0])[inverse])

    def __get_new_column_names(self, column_name, categories):
        return [column_name + "_" + category for category in categories]

    def __create_columns(self, one_hot_array):
        return [NumericColumn(one_hot_array[:, index]) for index in range(one_hot_array.shape[1])]
