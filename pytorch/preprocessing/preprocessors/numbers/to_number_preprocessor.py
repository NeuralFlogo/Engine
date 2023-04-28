class ToNumberPreprocessor:
    @staticmethod
    def process(col):
        return col.astype(int)