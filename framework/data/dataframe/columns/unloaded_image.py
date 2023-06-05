from framework.data.dataframe.column import Column


class UnloadedImageColumn(Column):
    def __init__(self, values=list()):
        self.values = [] if len(values) == 0 else values
        super(UnloadedImageColumn, self).__init__(values)
