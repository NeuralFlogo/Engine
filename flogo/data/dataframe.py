class Dataframe:
    def __init__(self, columns: dict = {}):
        self.columns = columns

    def __len__(self):
        return len(self.columns)

    def get(self, key):
        return self.columns[key]

    def append_columns(self, keys, columns):
        for index, key in enumerate(keys):
            self.append_column(key, columns[index])

    def append_column(self, key, column):
        self.columns[key] = column

    def get_column_names(self):
        return list(self.columns.keys())

    def update(self, dataframe):
        self.columns.update(dataframe.columns)

    def column_size(self):
        return len(list(self.columns.values())[0])
