from framework.data.dataframe.dataframe import Dataframe
from framework.preprocessing.delete_column import DeleteOperator
from framework.preprocessing.mapper import Mapper


class Orchestrator:
    def __init__(self, *mappers: Mapper):
        self.mappers = mappers
        self.delete_operator = DeleteOperator()

    def process(self, dataframe: Dataframe, *indexes):
        for index, mapper in enumerate(self.mappers):
            dataframe = mapper.map(dataframe, indexes[index])
            dataframe = self.delete_operator.delete(dataframe, indexes[index])
        return dataframe
