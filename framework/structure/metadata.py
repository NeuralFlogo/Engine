class Metadata:
    def __init__(self):
        self.start_indexes = []
        self.sections_length = []

    def add(self, start_index, section_length):
        self.start_indexes.append(start_index)
        self.sections_length.append(section_length)

    def get_start_index(self, index):
        return self.start_indexes[index]

    def get_sections_length(self, index):
        return self.sections_length[index]

    def get_end_index(self, index):
        return self.start_indexes[index] + self.sections_length[index]
