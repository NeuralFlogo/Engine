from pickle import dump


class ModelPersister:
    def persist(self, model, path):
        with open(path, "wb") as f:
            dump(model, f)
