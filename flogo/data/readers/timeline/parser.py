from flogo.data.readers.timeline.model.measurements import Measurements
from flogo.data.readers.timeline.model.id import Id
from flogo.data.readers.timeline.model.instant import Instant
from flogo.data.readers.timeline.model.measurements_list import MeasurementsList
from flogo.data.readers.timeline.model.period import Period


class Parser:
    def parse(self, line: str):
        return self.__switch(line[1:])

    def __switch(self, line):
        if line.startswith("id"): return Id(line[line.find(" ") + 1:])
        if line.startswith("instant"): return Instant(line[line.find(" ") + 1:-1])
        if line.startswith("period"): return Period(int(line.rstrip().split(" ")[1]), line.rstrip().split(" ")[2])
        if line.startswith("measurements"): return MeasurementsList([Measurements().set_values_with(measurement)
                                                                     for measurement in line[line.find(" ")+1:].split(",")])



