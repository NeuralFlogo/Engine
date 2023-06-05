from framework.data.timeline.id import Id
from framework.data.timeline.period import Period
from framework.data.timeline.timeline import Timeline
from framework.data.timeline.ts import Ts


class TimelineBuilder:
    def __init__(self):
        self.id = None
        self.ts = []
        self.period = None
        self.timeseries = None
        self.last_instant = None

    def set(self, line):
        for index, value in enumerate(line.split("\t")): self.timeseries[index].append(float(value))
        self.update()

    def set_command(self, param):
        if type(param) == Ts: self.last_instant = param.get_value()
        if type(param) == Period: self.period = param
        if type(param) == list: self.timeseries = param
        if type(param) == Id: self.id = param

    def update(self):
        self.ts.append(self.last_instant)
        self.last_instant += self.period.get_change()

    def build(self):
        for timeserie in self.timeseries: timeserie.set_ts(self.ts)
        return Timeline(self.id, self.ts, self.timeseries)
