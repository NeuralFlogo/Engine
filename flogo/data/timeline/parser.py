from flogo.data.timeline.id import Id
from flogo.data.timeline.period import Period
from flogo.data.timeline.timeserie import Timeserie
from flogo.data.timeline.ts import Ts


class Parser:
    def parse(self, line: str):
        return self.__switch(line[1:])

    def __switch(self, line):
        if line.startswith("id"): return Id(line[line.find(" ") + 1:])
        if line.startswith("instant"): return Ts(line[line.find(" ") + 1:-1])
        if line.startswith("period"): return Period(int(line.rstrip().split(" ")[1]), line.rstrip().split(" ")[2])
        if line.startswith("measurements"): return [Timeserie().set_values_with(measurement)
                                                    for measurement in line[line.find(" ")+1:].split(",")]
