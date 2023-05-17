from abc import ABC as Abstract
from abc import abstractmethod

from flogo.data.dataframe.column import Column
from flogo.data.dataframe.dataframe import Dataframe


class Mapper(Abstract):
    def map(self, input: Dataframe, indexes):
        output = Dataframe()
        for index in indexes:
            output.append_column(index + "'", self.apply(input.get(index)))
        output.update(input)
        return output

    @abstractmethod
    def apply(self, column: Column):
        pass
