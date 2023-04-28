import numpy as np


class NumericTransformer:
    def __init__(self, reader, preprocessor_list, shuffle: bool):
        self.data = reader.read()
        self.preprocessor_list = preprocessor_list
        self.inputs = self.__get_inputs()
        self.outputs = self.__get_outputs()
        if shuffle:
            self.__shuffle()

    def __get_inputs(self):
        inputs = self.data[:, 1:]
        transform_data = []
        for col in range(inputs.shape[1]):
            transform_data.append(self.preprocessor_list[col].process(inputs[:, col]))
        return np.concatenate(transform_data, axis=1)

    def __get_outputs(self):
        output_array = self.preprocessor_list[-1].process(self.data[:, 0])
        return output_array

    def __shuffle(self):
        index = np.random.permutation(len(self.outputs))
        self.inputs, self.outputs = self.inputs[index], self.outputs[index].tolist()

    def transform_inputs(self):
        return self.inputs

    def transform_outputs(self):
        return self.outputs
