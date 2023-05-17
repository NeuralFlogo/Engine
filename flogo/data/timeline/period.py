from flogo.data.timeline.utils.metrics import *


class Period:
    def __init__(self, measure, magnitude):
        self.measure = measure
        self.magnitude = self.unit(magnitude)
        self.change = self.__set_change()

    def unit(self, unit):
        magnitude = self.__switch(unit)
        if magnitude is not None: return magnitude
        raise RuntimeError("Inconsistent period unit")

    def __switch(self, unit):
        if unit == "seconds": return SECOND
        if unit == "minutes": return MINUTE
        if unit == "hours": return HOUR
        if unit == "days": return DAY
        if unit == "weeks": return WEEK
        if unit == "month": return MONTH
        if unit == "quarters": return QUARTER
        if unit == "halfyears": return HALF_YEAR
        if unit == "years": return YEAR

    def get_value(self):
        return self.measure

    def get_measure(self):
        return self.magnitude

    def get_change(self):
        return self.change

    def __set_change(self):
        return self.measure * self.magnitude

    def set_measure(self, measure):
        self.measure = measure
