import numpy as np


class OneHotPreprocessor:
    @staticmethod
    def process(col):
        unique, inverse = np.unique(col, return_inverse=True)
        return np.eye(unique.shape[0])[inverse]