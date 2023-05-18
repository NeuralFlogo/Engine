from framework.data.dataframe.columns.number import NumericColumn
from framework.data.dataframe.dataframe import Dataframe
from framework.data.timeline.timeserie import Timeserie


class Timeline:
    def __init__(self, id, timestamps, timeseries: list):
        self.id = id
        self.timestamps = timestamps
        self.timeseries = timeseries
        self.normalized_instants = None

    def __len__(self):
        return len(self.timestamps)

    def get_normalized_instants(self):
        if self.normalized_instants is None:
            initial = self.timestamps[0]
            self.normalized_instants = [instant - initial for instant in self.timestamps]
        return self.normalized_instants

    def group_by(self, quantity, magnitude):
        boundary_indexes = self.__get_boundary_indexes(quantity, magnitude)
        instants = self.__get_timestamps(boundary_indexes)
        return Timeline(self.id, instants, self.__create_timeseries(boundary_indexes, instants))

    def to_dataframe(self, window=5, offset=0):
        dataframe = Dataframe()
        for timeserie in self.timeseries:
            input_columns, output_column = self.__create_columns(timeserie, window, offset)
            dataframe.append_columns(self.__create_keys(timeserie.name, window), input_columns + output_column)
        return dataframe

    def __get_boundary_indexes(self, quantity, magnitude):
        indexes, update_value = [0], quantity
        for index, value in enumerate(self.get_normalized_instants()):
            if value >= quantity * magnitude:
                quantity += update_value
                indexes.append(index)
        indexes.append(len(self) - 1)
        return indexes

    def __get_timestamps(self, boundary_indexes):
        return [self.timestamps[index] for index in boundary_indexes]

    def __create_timeseries(self, boundary_indexes, timestamps):
        return [self.__create_timeserie(timeserie, boundary_indexes, timestamps) for timeserie in self.timeseries]

    def __create_timeserie(self, timeserie, boundary_indexes, timestamps):
        return Timeserie(self.__get_values(boundary_indexes, timeserie), timeserie.name, timeserie.direction,
                         timeserie.operator, timestamps)

    def __get_values(self, boundary_indexes, timeserie):
        return [timeserie.operator(timeserie.get_values()[boundary_indexes[index] - 1:boundary_indexes[index]])
                for index in range(1, len(boundary_indexes))]

    def __create_columns(self, timeserie, window, offset):
        input_columns, output_column = [NumericColumn() for _ in range(window)], NumericColumn()
        for output_value_index in range(window, len(timeserie) - offset):
            for index, value in enumerate(timeserie.get_values()[output_value_index - window:output_value_index]): input_columns[index].append(value)
            output_column.append(timeserie.get_values()[output_value_index + offset])
        return input_columns, [output_column]

    def __create_keys(self, timeserie_name, window):
        return [timeserie_name + "_input_" + str(i) for i in range(window)] + [timeserie_name + "_output"]
