from framework.data.dataframe.column import Column


class CategoricalColumn(Column):
    def __init__(self, values=list()):
        self.values = [] if len(values) == 0 else values
        super(CategoricalColumn, self).__init__(self.values)
