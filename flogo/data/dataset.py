class Dataset:

    def __init__(self, entries: list = []):
        self.entries = entries

    def get_entries(self):
        return self.entries

    def __len__(self):
        return self.entries

    def __getitem__(self, idx):
        return self.entries[idx]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index == len(self):
            raise StopIteration
        return self.entries[self.index]
