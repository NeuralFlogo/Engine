class Dataset:
    def __init__(self, entries: list = []):
        self.entries = entries
        self.size = self.__compute_size()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.entries[idx]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index == self.batch_count():
            raise StopIteration
        entry = self.entries[self.index]
        self.index += 1
        return entry

    def __compute_size(self):
        return sum([entry.get_size() for entry in self.entries])

    def get_entries(self):
        return self.entries

    def batch_count(self):
        return len(self.entries)
