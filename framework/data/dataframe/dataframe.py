class Dataframe:
    def __init__(self, columns: dict = {}):
        self.columns = columns

    def __len__(self):
        return len(self.columns)

    def column_size(self):
        return len(list(self.columns.values())[0])

    def get(self, key):
        return self.columns[key]

    def column_names(self):
        return list(self.columns.keys())

    def append_column(self, key, column):
        self.columns[key] = column

    def append_columns(self, keys, columns):
        for index, key in enumerate(keys):
            self.append_column(key, columns[index])

    def update(self, dataframe):
        self.columns.update(dataframe.columns)
