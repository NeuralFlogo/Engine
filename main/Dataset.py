from torch.utils.data import IterableDataset


class Dataset(IterableDataset):

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)