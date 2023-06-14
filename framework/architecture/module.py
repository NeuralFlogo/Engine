class Module:
    def __init__(self, section, length):
        self.section = section
        self.length = length

    def __len__(self):
        return self.length

    def get_section(self):
        return self.section

    def get_len(self):
        return self.length
