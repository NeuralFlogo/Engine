import random

from flogo.data.dataset import Dataset


class DatasetSplitter:

    def split(self, dataset: Dataset, test_proportion=0.2, validation_proportion=0.16, shuffle: bool = True):
        test_index, validation_index = self.__get_indexes(dataset, test_proportion, validation_proportion)
        entries = self.__shuffle(dataset.get_entries()) if shuffle else dataset.get_entries()
        return Dataset(entries[:test_index]), Dataset(entries[test_index:validation_index]), Dataset(entries[validation_index:])

    def __get_indexes(self, dataset, test_proportion, validation_proportion):
        validation_index = self.__index(len(dataset), validation_proportion, len(dataset))
        return self.__index(validation_index, test_proportion, len(dataset)), validation_index

    @staticmethod
    def __index(end, proportion, dataset_size):
        return round(end - (proportion * dataset_size))

    @staticmethod
    def __shuffle(entries):
        random.shuffle(entries)
        return entries
