from pickle import load


class ModelLoader:
    def __init__(self, path):
        self.path = path
        self.model = None

    def load(self):
        if self.model is None:
            with open(self.path, "rb") as f: model = load(f)
        return model

    def load_section(self, index):
        return self.load().get_section(index)
