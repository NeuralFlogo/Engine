from pytorch.preprocesing.NumericProcessor import *

transformer_functions = {
    "one-hot": one_hot_encode,
    "standardize": standardize,
    "normalize": normalize,
    "to-number": to_number
}


class NumericNumpyTransformer:
    def __init__(self, data, parameters, boolean_shufle):
        self.data = np.array(data)
        if boolean_shufle:
            np.random.shuffle(self.data)
        self.parameters = parameters

    def transform_inputs(self):
        inputs = self.data[:, 1:-1]
        transform_data = []
        for col in range(inputs.shape[1]):
            transform_data.append(transformer_functions.get(self.parameters[col])((inputs[:, col])))
        return np.concatenate(transform_data, axis=1).tolist()

    def transform_outputs(self):
        output_array = transformer_functions.get(self.parameters[-1])(self.data[:, 0])
        return output_array.tolist()



