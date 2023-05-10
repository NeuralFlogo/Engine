import random

from flogo.data.dataset import Dataset


class DatasetSplitter:

    def split(self, dataset: Dataset, test_proportion=0.2, validation_proportion=0.16, shuffle: bool = True):
        test_boundary, validation_boundary = self.__boundaries(dataset, test_proportion, validation_proportion)
        entries = self.__shuffle(dataset.get_entries()) if shuffle else dataset.get_entries()
        return Dataset(entries[:test_boundary]), Dataset(entries[test_boundary:validation_boundary]), Dataset(entries[validation_boundary:])

    def __boundaries(self, dataset, test_proportion, validation_proportion):
        validation_index = self.__index(dataset.batch_count(), validation_proportion, dataset.batch_count())
        return self.__index(validation_index, test_proportion, dataset.batch_count()), validation_index

    def __index(self, end, proportion, dataset_size):
        return round(end - (proportion * dataset_size))

    def __shuffle(self, entries):
        random.shuffle(entries)
        return entries
