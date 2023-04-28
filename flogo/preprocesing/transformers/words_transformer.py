class WordsTransformer:
    def __init__(self, string, preprocessor):
        self.tokens = preprocessor.process(string)

    def __len__(self):
        return len(self.tokens)

    def __iter__(self, idx):
        return self.tokens[idx]
