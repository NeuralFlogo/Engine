from flogo.data.columns.number import NumericColumn
from flogo.data.dataframe import Dataframe
from flogo.data.readers.timeline.parser import MeasurementsList, Measurements


class Timeline:
    def __init__(self, id, instants, measurements_list: MeasurementsList):
        self.id = id
        self.instants = instants
        self.measurements_list = measurements_list
        self.normalized_instants = None

    def __len__(self):
        return len(self.instants)

    def get_normalized_instants(self):
        if self.normalized_instants is None:
            initial = self.instants[0]
            self.normalized_instants = [instant - initial for instant in self.instants]
        return self.normalized_instants

    def group_by(self, quantity, magnitude):
        indexes = self.__get_indexes(quantity, magnitude)
        return Timeline(self.id, self.__create_grouped_indexes(indexes), self.__create_grouped_measurements(indexes))

    def __get_indexes(self, quantity, magnitude):
        indexes, update_value = [0], quantity
        for index, value in enumerate(self.get_normalized_instants()):
            if value >= quantity * magnitude:
                quantity += update_value
                indexes.append(index)
        indexes.append(len(self) - 1)
        return indexes

    def __create_grouped_indexes(self, indexes):
        return [self.instants[index] for index in indexes]

    def __create_grouped_measurements(self, indexes):
        return MeasurementsList([self.__create_measurements(measurements, indexes) for index, measurements
                                 in enumerate(self.measurements_list.get_measurements())])

    def __create_measurements(self, measurements, indexes):
        return Measurements([measurements.operator(self.__get_values(index, indexes, measurements))
                             for index in range(1, len(indexes))], measurements.name, measurements.direction,
                             measurements.operator)

    def __get_values(self, index, indexes, measurements):
        return measurements.get_values()[indexes[index - 1]:indexes[index]]

    def to_dataframe(self, window=5):
        dataframe = Dataframe()
        for measurements in self.measurements_list.get_measurements():
            input_columns, output_columns = self.__create_columns(measurements, window)
            dataframe.append_columns(self.__create_keys(measurements.name, window), output_columns + input_columns)
        return dataframe

    def __create_columns(self, measurements, window):
        input_columns, output_column = [NumericColumn() for _ in range(window)], NumericColumn()
        for label_index in range(window, len(measurements)):
            for index, value in enumerate(measurements.get_values()[label_index - window:label_index]):
                input_columns[index].append(value)
            output_column.append(measurements.get_values()[label_index])
        return input_columns, [output_column]

    def __create_keys(self, measurements_name, window):
        return [measurements_name + "_output"] + [measurements_name + "_input_" + str(i) for i in range(window)]
