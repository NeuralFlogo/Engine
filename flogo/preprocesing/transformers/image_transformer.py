class ImageTransformer:
    def __init__(self, reader, preprocessor, shuffle):
        data_path = reader.download()
        inputs, labels = preprocessor.process(data_path, shuffle)
        self.inputs = inputs
        self.labels = labels

    def transform_inputs(self):
        return self.inputs

    def transform_outputs(self):
        return self.labels
