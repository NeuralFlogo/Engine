from flogo.preprocessing.column import Column


class Dataframe:
    def __init__(self, columns: dict = {}):
        self.columns = columns

    def __len__(self):
        return len(self.columns)

    def append(self, key, columns):
        self.append_column(key + "'", columns) if type(columns) != list else self.append_list(key, columns)

    def get(self, key):
        return self.columns[key]

    def update(self, dataframe):
        self.columns.update(dataframe.columns)

    def append_column(self, key, column):
        self.columns[key] = column

    def append_list(self, key, columns):
        for index, column in enumerate(columns):
            self.append_column(key + str(index), column)


