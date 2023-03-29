import torch
import torch.nn.functional as f


def encode(col):
    return f.one_hot(col[:, :-1], torch.unique(col[:, :-1]).len())


def standardize(col):
    return (col - torch.mean(col)) / torch.std(col)


def normalize(col, new_min, new_max):
    return (col - col.min()) / (col.max() - col.min()) * (new_max - new_min) + new_min


def l_normalize(col, p, dim=1):
    return f.normalize(col, p, dim)
