from torch.utils.data import Dataset

from pytorch.preprocesing.TextTokenizer import tokenize


class WordsTransformer(Dataset):
    def __init__(self, string):
        self.tokens = tokenize(string)

    def __len__(self):
        return len(self.tokens)

    def __iter__(self, idx):
        return self.tokens[idx]
