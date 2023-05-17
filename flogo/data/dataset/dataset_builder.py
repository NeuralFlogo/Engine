from flogo.data.dataframe.dataframe import Dataframe
from flogo.data.dataset.dataset import Dataset
from flogo.data.dataset.entry import Entry


class DatasetBuilder:
    def __init__(self, caster):
        self.caster = caster

    def build(self, dataframe: Dataframe, input_columns: list, output_columns: list, batch_size: int = 1) -> Dataset:
        entries = []
        for index in range(0, dataframe.column_size(), batch_size):
            entries.append(self.__create_entry(dataframe, index, input_columns, output_columns, batch_size))
        return Dataset(entries)

    def __create_entry(self, dataframe, index, input_columns, output_columns, batch_size):
        return Entry(self.__entry_size(batch_size, dataframe, index),
                     self.__get_values(dataframe, index, input_columns, batch_size),
                     self.__get_values(dataframe, index, output_columns, batch_size))

    def __entry_size(self, batch_size, dataframe, index):
        return batch_size if index + batch_size <= dataframe.column_size() else dataframe.column_size() - index

    def __get_values(self, dataframe, index, column_names, batch_size):
        end = index + batch_size
        if end > dataframe.column_size():
            end = dataframe.column_size()
        return self.__batch(dataframe, index, end, column_names)

    def __batch(self, dataframe, start, end, columns):
        batches = []
        for index in range(start, end):
            batch = self.__group_by_row(dataframe, columns, index)
            batches.append(batch) if len(columns) != 1 else batches.extend(batch)
        return self.caster.cast(batches)

    def __group_by_row(self, dataframe, columns, index):
        return [dataframe.get(column).get_values()[index] for column in columns]
