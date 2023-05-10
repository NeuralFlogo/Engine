class Column:

    def __init__(self, values):
        self.values = values

    def append(self, value):
        self.values.append(value)

    def __len__(self):
        return len(self.values)

    def get(self, idx):
        return self.values[idx]

    def get_values(self):
        return self.values
