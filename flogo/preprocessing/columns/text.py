from flogo.preprocessing.column import Column


class TextColumn(Column):

    def __init__(self, values=list()):
        self.values = [] if len(values) == 0 else values
        super(TextColumn, self).__init__(values)
