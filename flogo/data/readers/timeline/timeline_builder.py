from flogo.data.readers.timeline.parser import Instant, MeasurementsList, Period, Id
from flogo.data.readers.timeline.timeline import Timeline


class TimelineBuilder:
    def __init__(self):
        self.id = None
        self.instants = []
        self.period = None
        self.measurements = None
        self.last_instant = None

    def set(self, line):
        for index, value in enumerate(line.split("\t")): self.measurements.get(index).append(float(value))
        self.update()

    def set_command(self, param):
        if type(param) == Instant: self.last_instant = param.get_value() - self.period.get_change()
        if type(param) == Period: self.period = param
        if type(param) == MeasurementsList: self.measurements = param
        if type(param) == Id: self.id = param

    def update(self):
        self.last_instant += self.period.get_change()
        self.instants.append(self.last_instant)

    def build(self):
        return Timeline(self.id, self.instants, self.measurements)
