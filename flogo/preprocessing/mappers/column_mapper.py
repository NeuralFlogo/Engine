import numpy as np

from flogo.preprocessing.columns.number import NumericColumn
from flogo.preprocessing.mapper import Mapper


class ColumnMapper(Mapper):

    def __init__(self, column_type):
        self.column_type = column_type

    def apply(self, column: NumericColumn):
        return self.column_type(np.array(column.values).tolist())

