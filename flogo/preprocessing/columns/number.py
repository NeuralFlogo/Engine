from flogo.preprocessing.column import Column


class NumericColumn(Column):

    def __init__(self, values=[], dtype=int):
        self.dtype = dtype
        super(NumericColumn, self).__init__(self.convert(values))

    def convert(self, values):
        return list(map(self.dtype, values))

    def append(self, value):
        self.values.append(self.dtype(value))