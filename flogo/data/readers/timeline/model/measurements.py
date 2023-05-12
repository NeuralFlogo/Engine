import statistics


class Measurements:
    def __init__(self, values=list(), name=None, direction=None, operator=None):
        self.name = name
        self.direction = direction
        self.operator = operator
        self.values = [] if len(values) == 0 else values

    def set_values_with(self, line):
        line = line.split(":")
        self.name = line[0]
        self.direction = line[1][9:] if len(line) > 1 else None
        self.operator = self.set_operator(line[2][9:]) if len(line) > 2 else sum
        return self

    def append(self, value):
        self.values.append(value)

    def get_values(self):
        return self.values

    def __len__(self):
        return len(self.values)

    def set_operator(self, operator: str):
        if operator == "average": return statistics.mean
