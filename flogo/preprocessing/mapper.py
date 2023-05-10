from abc import ABC as Abstract
from abc import abstractmethod

from flogo.data.column import Column
from flogo.data.dataframe import Dataframe


class Mapper(Abstract):

    def map(self, dataframe: Dataframe, indexes):
        result = Dataframe()
        for index in indexes:
            result.append_column(index + "'", self.apply(dataframe.get(index)))
        result.update(dataframe)
        return result

    @abstractmethod
    def apply(self, column: Column):
        pass
