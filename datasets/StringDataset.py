from torch.utils.data import Dataset

from preprocesing.TextTokenizer import tokenize


def tokens(paths):
    text = ""
    for path in paths:
        with open(path, 'r') as file:
            text += file.read().replace('\n', ' ') + '\n'
    return tokenize(text)


class StringDataset(Dataset):
    def __init__(self, paths):
        self.tokens = tokens(paths)

    def __len__(self):
        return len(self.tokens)

    def __iter__(self, idx):
        return self.tokens[idx]
