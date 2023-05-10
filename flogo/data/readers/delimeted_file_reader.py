from flogo.data.dataframe import Dataframe


class DelimitedFileReader:

    def __init__(self, delimiter):
        self.delimiter = delimiter

    def read(self, path, columns: dict, header: bool = True):
        with open(path, "r") as file:
            self.__fill_columns_with(file.read(), columns, header)
        return Dataframe(columns)

    def __fill_columns_with(self, content, columns, header):
        for line in self.__lines(content, header):
            self.__fill(columns, line)

    def __lines(self, content, header):
        return content.split("\n")[1:] if header else content.split("\n")

    def __fill(self, columns, line):
        for index, value in enumerate(line.split(self.delimiter)):
            column = columns[list(columns.keys())[index]]
            column.append(value)
