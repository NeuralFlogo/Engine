from flogo.preprocessing.dataframe import Dataframe


class DelimitedFileReader:

    def __init__(self, delimiter):
        self.delimiter = delimiter

    def read(self, path, columns: dict, header: bool = True):
        with open(path, "r") as file:
            self.__read_values(file.read(), columns, header)
        return Dataframe(columns)

    def __read_values(self, content, columns, header):
        for line in self.__get_lines(content, header):
            self.__get_values_from(line, columns)

    @staticmethod
    def __get_lines(content, header):
        return content.split("\n")[1:] if header else content.split("\n")

    def __get_values_from(self, line, columns):
        for index, value in enumerate(line.split(self.delimiter)):
            column = columns[list(columns.keys())[index]]
            column.append(value)
