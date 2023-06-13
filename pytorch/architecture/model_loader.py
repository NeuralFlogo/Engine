from pickle import load


class ModelLoader:
    def load(self, path):
        with open(path, "rb") as f:
            model = load(f)
        return model
