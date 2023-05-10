class Column:

    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def get(self, index):
        return self.values[index]

    def get_values(self):
        return self.values

    def append(self, value):
        self.values.append(value)

