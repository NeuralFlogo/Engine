import numpy as np


class StandarizationPreprocessor:
    @staticmethod
    def process(col):
        col = col.astype(float)
        return (col - np.mean(col)) / np.std(col).reshape(-1, 1)