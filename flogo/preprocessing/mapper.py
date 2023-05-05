from abc import ABC as Abstract
from abc import abstractmethod

from flogo.preprocessing.column import Column
from flogo.preprocessing.dataframe import Dataframe


class Mapper(Abstract):

    def process(self, dataframe: Dataframe, indexes):
        result = Dataframe()
        for index in indexes:
            result.append(index, self.apply(dataframe.get(index)))
        result.update(dataframe)
        return result

    @abstractmethod
    def apply(self, column: Column):
        pass
