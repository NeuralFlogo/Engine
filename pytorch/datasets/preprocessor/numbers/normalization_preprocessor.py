class NormalizationPreprocessor:

    def __init__(self, min=0, max=1):
        self.min = min
        self.max = max

    def process(self, col):
        col = col.astype(float)
        return ((col - col.min()) / (col.max() - col.min()) * (self.max - self.min) + self.min).reshape(-1, 1)