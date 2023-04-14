import numpy as np


def one_hot_encode(col):
    unique, inverse = np.unique(col, return_inverse=True)
    return np.eye(unique.shape[0])[inverse]


def standardize(col):
    col = col.astype(float)
    return (col - np.mean(col)) / np.std(col).reshape(-1, 1)


def normalize(col, new_min=0, new_max=1):
    col = col.astype(float)
    return ((col - col.min()) / (col.max() - col.min()) * (new_max - new_min) + new_min).reshape(-1, 1)


def to_number(col):
    return col.astype(int)
