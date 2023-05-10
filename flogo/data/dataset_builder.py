from flogo.data.dataframe import Dataframe
from flogo.data.dataset import Dataset
from flogo.data.entry import Entry


class DatasetBuilder:

    def __init__(self, caster):
        self.caster = caster

    def build(self, dataframe: Dataframe, input_columns: list, output_columns: list, batch_size: int = 1) -> Dataset:
        entries = []
        for index in range(0, dataframe.columns_len(), batch_size):
            entries.append(self.__create_entry(dataframe, index, input_columns, output_columns, batch_size))
        return Dataset(entries)

    def __create_entry(self, dataframe, index, input_columns, output_columns, batch_size):
        return Entry(self.__get_values(dataframe, index, input_columns, batch_size),
                     self.__get_values(dataframe, index, output_columns, batch_size))

    def __get_values(self, dataframe, index, names, batch_size):
        end = index + batch_size
        if end > dataframe.columns_len(): end = -1
        print(dataframe.get("input'''").values[index:end])
        return self.caster.cast([dataframe.get(column).values[index:end] for column in names])
