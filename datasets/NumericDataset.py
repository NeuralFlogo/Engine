from torch.utils.data import Dataset

from preprocesing.NumericProcessor import *

transformer_functions = {
    "one-hot": one_hot_encode,
    "standardize": standardize,
    "normalize": normalize

}


def transform(data, parameters):
    transform_data = []
    for col in range(data.shape[1]):
        transform_data.append(transformer_functions.get(parameters[col])((data[:, col])))
    return np.concatenate(transform_data, axis=1)


class NumericDataset(Dataset):
    def __init__(self, x, y, parameters):
        self.x = transform(x, parameters)
        self.y = transformer_functions.get(parameters[-1])(y)

    def __len__(self):
        return len(self.y)

    def __iter__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y
