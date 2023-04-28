import numpy as np


class FileReader:

    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        with open(self.file_path, "r") as archivo:
            return self.__get_array_from(archivo.read())

    def download(self):
        return self.file_path

    @staticmethod
    def __get_array_from(data_string):
        data_list = [x.split(',') for x in data_string.split('\n')]
        data_list.pop()
        return np.array(data_list)