from framework.data.dataframe.columns.number import NumericColumn
from framework.preprocessing.mapper import Mapper


class TypeMapper(Mapper):
    def __init__(self, column_type):
        self.column_type = column_type

    def apply(self, column: NumericColumn):
        return self.column_type(column.values)

