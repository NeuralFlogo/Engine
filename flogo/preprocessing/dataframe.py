class Dataframe:
    def __init__(self, columns: dict = {}):
        self.columns = columns

    def __len__(self):
        return len(self.columns)

    def get(self, key):
        return self.columns[key]

    def append_column(self, key, column):
        self.columns[key] = column

    def get_column_names(self):
        return list(self.columns.keys())

    def update(self, dataframe):
        self.columns.update(dataframe.columns)

    def append_updated_columns(self, key, columns):
        self.append_column(key + "'", columns) if type(columns) != list else self.__append_columns(key, columns)

    def __append_columns(self, key, columns):
        for index, column in enumerate(columns):
            self.append_column(key + str(index), column)
