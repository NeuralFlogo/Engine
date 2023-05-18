import statistics


class Timeserie:
    def __init__(self, values=list(), name=None, direction=None, operator=None, timestamps=None):
        self.name = name
        self.direction = direction
        self.operator = operator
        self.measurements = [] if len(values) == 0 else values
        self.timestamps = timestamps

    def __len__(self):
        return len(self.measurements)

    def get_values(self):
        return self.measurements

    def set_values_with(self, line):
        line = line.split(":")
        self.name = line[0]
        self.direction = line[1][9:] if len(line) > 1 else None
        self.operator = self.set_operator(line[2][9:]) if len(line) > 2 else sum
        return self

    def set_operator(self, operator: str):
        if operator == "average": return statistics.mean

    def set_ts(self, ts):
        self.timestamps = ts

    def append(self, value):
        self.measurements.append(value)
